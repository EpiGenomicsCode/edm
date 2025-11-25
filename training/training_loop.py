# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import sys
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

# Validation hook.
from validation import maybe_validate, run_fid_validation

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    wandb_kwargs        = None,     # Options for Weights & Biases logging, None = disable.
    wandb_config        = None,     # Serializable run config to send to W&B (optional).
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    validation_kwargs   = None,     # Validation configuration (PRD-04).
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    # Debug + optional init-from-teacher when doing consistency distillation.
    if hasattr(loss_fn, 'teacher_net'):
        teacher = loss_fn.teacher_net
        # Shape diagnostics on rank 0.
        if dist.get_rank() == 0:
            try:
                dist.print0('[CD DEBUG] Teacher vs Student diagnostic:')
                dist.print0(f'[CD DEBUG]   teacher class: {type(teacher).__name__}')
                dist.print0(f'[CD DEBUG]   student class: {type(net).__name__}')
                t_label_dim = getattr(teacher, 'label_dim', None)
                s_label_dim = getattr(net, 'label_dim', None)
                dist.print0(f'[CD DEBUG]   teacher label_dim: {t_label_dim}')
                dist.print0(f'[CD DEBUG]   student label_dim: {s_label_dim}')
                # Grab a likely label-related param name and shape.
                def _first_label_param(mod):
                    for n, p in mod.named_parameters():
                        if 'label' in n or 'map_label' in n:
                            return n, tuple(p.shape)
                    return None, None
                t_lp_name, t_lp_shape = _first_label_param(teacher)
                s_lp_name, s_lp_shape = _first_label_param(net)
                dist.print0(f'[CD DEBUG]   teacher label param: {t_lp_name} {t_lp_shape}')
                dist.print0(f'[CD DEBUG]   student label param: {s_lp_name} {s_lp_shape}')
            except Exception as _e:
                dist.print0(f'[CD DEBUG]   diagnostics failed: {_e}')

        # Optionally seed student weights from teacher when shapes match exactly.
        try:
            def _shapes(mod):
                d = {}
                for n, p in mod.named_parameters():
                    d[n] = tuple(p.shape)
                for n, b in mod.named_buffers():
                    d[n] = tuple(b.shape)
                return d
            t_shapes = _shapes(teacher)
            s_shapes = _shapes(net)
            mismatches = []
            for name, tshape in t_shapes.items():
                sshape = s_shapes.get(name)
                if sshape is None or sshape != tshape:
                    mismatches.append((name, tshape, sshape))
            if len(mismatches) == 0:
                misc.copy_params_and_buffers(src_module=teacher, dst_module=net, require_all=False)
                if dist.get_rank() == 0:
                    dist.print0('[CD INIT] Seeded student weights from teacher (all parameter/buffer shapes match).')
            else:
                if dist.get_rank() == 0:
                    dist.print0(f'[CD INIT] Not seeding from teacher because {len(mismatches)} tensors differ in shape.')
        except Exception as _e:
            dist.print0(f'[CD INIT] init-from-teacher failed: {_e}')

    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    # Use static_graph=True to avoid DDP bucket rebuilds mid-training (which have
    # been triggering negative-dimension errors in reducer._rebuild_buckets) and
    # disable broadcast_buffers since we only use GroupNorm (no running stats).
    ddp = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[device],
        broadcast_buffers=False,
        static_graph=True,
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    
    # Seed student from teacher AFTER DDP wrapping (if CD mode and shapes match).
    # This avoids NCCL desync issues during DDP initialization.
    if hasattr(loss_fn, 'teacher_net'):
        try:
            teacher = loss_fn.teacher_net
            # Check if all params/buffers match in shape.
            def _shapes(mod):
                d = {}
                for n, p in mod.named_parameters():
                    d[n] = tuple(p.shape)
                for n, b in mod.named_buffers():
                    d[n] = tuple(b.shape)
                return d
            t_shapes = _shapes(teacher)
            s_shapes = _shapes(net)
            mismatches = []
            for name, tshape in t_shapes.items():
                sshape = s_shapes.get(name)
                if sshape is None or sshape != tshape:
                    mismatches.append((name, tshape, sshape))
            if len(mismatches) == 0:
                # All shapes match; copy weights to the underlying net module.
                misc.copy_params_and_buffers(src_module=teacher, dst_module=net, require_all=False)
                # Also update EMA to start from teacher weights.
                misc.copy_params_and_buffers(src_module=teacher, dst_module=ema, require_all=False)
                if dist.get_rank() == 0:
                    dist.print0('[CD INIT] Seeded student & EMA from teacher (all parameter/buffer shapes match).')
            else:
                if dist.get_rank() == 0:
                    dist.print0(f'[CD INIT] Not seeding from teacher because {len(mismatches)} tensors differ in shape.')
        except Exception as _e:
            if dist.get_rank() == 0:
                dist.print0(f'[CD INIT] Failed to seed from teacher: {_e}')

    # Optional W&B initialization (async/threaded).
    wandb_run = None
    if wandb_kwargs is not None and wandb_kwargs.get('enabled', False) and dist.get_rank() == 0:
        try:
            import wandb as _wandb
            # Make W&B robust with our stdout redirection: avoid isatty uses.
            # Provide isatty() when stdout/stderr are our custom Logger without it.
            try:
                if not hasattr(sys.stdout, 'isatty'):
                    sys.stdout.isatty = lambda: False
                if not hasattr(sys.stderr, 'isatty'):
                    sys.stderr.isatty = lambda: False
            except Exception:
                pass
            init_kwargs = dict(
                project=wandb_kwargs.get('project', 'edm-consistency'),
                entity=wandb_kwargs.get('entity', None),
                name=wandb_kwargs.get('name', None),
                tags=wandb_kwargs.get('tags', None),
            )
            mode = wandb_kwargs.get('mode', 'online')
            if mode in ('offline', 'disabled'):
                init_kwargs['mode'] = mode
            # Use default settings so W&B can capture console logs.
            wandb_run = _wandb.init(**init_kwargs, config=wandb_config)
            # Also sync log.txt into the W&B run Files tab (live).
            try:
                _wandb.save(os.path.join(run_dir, 'log.txt'), policy='live')
            except Exception:
                pass
        except Exception as _e:
            dist.print0(f'[W&B] init failed: {_e}')
            wandb_run = None

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Broadcast run_dir to all ranks (rank 0 has it, others have None).
    if dist.get_rank() == 0:
        run_dir_str = run_dir if run_dir is not None else ''
    else:
        run_dir_str = ''
    # Use object list broadcast.
    run_dir_list = [run_dir_str]
    dist.ddp_debug(f'run_dir broadcast: before, run_dir_str="{run_dir_str}"')
    torch.distributed.broadcast_object_list(run_dir_list, src=0)
    run_dir = run_dir_list[0] if run_dir_list[0] else None
    dist.ddp_debug(f'run_dir broadcast: after, run_dir="{run_dir}"')
    
    # One-time teacher validation (baseline) using ImageNet defaults if available.
    try:
        if (validation_kwargs is not None
            and validation_kwargs.get('enabled', True)
            and validation_kwargs.get('teacher', True)
            and hasattr(loss_fn, 'teacher_net')):
            # Rank 0 decides; broadcast to all.
            should_run_teacher = False
            if dist.get_rank() == 0:
                teacher_done_flag = os.path.join(run_dir, 'val_teacher.json')
                should_run_teacher = not os.path.isfile(teacher_done_flag)
            # Broadcast decision from rank 0.
            flag_tensor = torch.tensor([1 if should_run_teacher else 0], dtype=torch.int64, device=device)
            dist.ddp_debug(f'teacher_flag broadcast: before, val={int(flag_tensor.item())}')
            torch.distributed.broadcast(flag_tensor, src=0)
            should_run_teacher = bool(flag_tensor.item())
            dist.ddp_debug(f'teacher_flag broadcast: after, val={int(flag_tensor.item())}')
            # Global sync so either all enter validation together or none do.
            torch.distributed.barrier()
            if should_run_teacher:
                # Use the same teacher object that the CD loss uses.
                teacher_net = loss_fn.teacher_net.eval().requires_grad_(False).to(device)
                # Teacher sampler defaults per README ImageNet.
                # IMPORTANT: Do NOT override sigma_min/sigma_max here — edm_sampler
                # has good defaults (0.002, 80) and clamps against net.sigma_min/max.
                # Passing net.sigma_min/max (=0, inf) would break the grid and cause NaNs.
                teacher_sampler = dict(
                    kind='edm',
                    num_steps=256,
                    rho=7.0,
                    S_churn=40, S_min=0.05, S_max=50.0, S_noise=1.003,
                )
                dist.print0('[VAL] Running one-time teacher validation (ImageNet defaults)...')
                teacher_dump_dir = os.path.join(run_dir, 'teacher_samples') if dist.get_rank() == 0 else None
                result = run_fid_validation(
                    teacher_net,
                    run_dir=run_dir,
                    dataset_kwargs=dataset_kwargs,
                    num_images=int(validation_kwargs.get('num_images', 50000)),
                    batch=int(validation_kwargs.get('batch', 64)),
                    seed=int(validation_kwargs.get('seed', 0)),
                    sampler=teacher_sampler,
                    labels='auto',
                    ref=validation_kwargs.get('ref', None),
                    ref_data=validation_kwargs.get('ref_data', None),
                    dump_images_dir=teacher_dump_dir,
                    overwrite=False,
                    step_kimg=None,
                    wandb_run=wandb_run,
                )
                if dist.get_rank() == 0:
                    payload = dict(fid=result.get('fid'), sampler=teacher_sampler)
                    with open(teacher_done_flag, 'wt') as f:
                        json.dump(payload, f)
    except Exception as _e:
        dist.print0(f'[VAL] teacher validation failed: {_e}')

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    ema_updates = 0
    last_loss_scalar = None
    while True:

        # Make teacher annealing resume-aware by exposing global kimg to the loss.
        try:
            if hasattr(loss_fn, 'set_global_kimg'):
                loss_fn.set_global_kimg(cur_nimg / 1e3)
        except Exception:
            pass

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                try:
                    last_loss_scalar = float(loss.mean().detach().cpu().item())
                except Exception:
                    pass

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        ema_updates += 1

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            # Optional W&B logging once per tick (async).
            if wandb_run is not None:
                try:
                    log_dict = dict(
                        progress_kimg=cur_nimg / 1e3,
                        tick=cur_tick,
                        loss=last_loss_scalar if last_loss_scalar is not None else None,
                        ema_updates=ema_updates,
                    )
                    # Teacher grid T edges if exposed by CD loss.
                    try:
                        if hasattr(loss_fn, '_current_T_edges'):
                            log_dict['cd_T_edges'] = int(loss_fn._current_T_edges())
                    except Exception:
                        pass
                    # Merge detailed training stats.
                    stats_payload = training_stats.default_collector.as_dict()
                    if isinstance(stats_payload, dict):
                        log_dict.update(stats_payload)
                    import wandb as _wandb
                    _wandb.log(log_dict, commit=True)
                except Exception as _e:
                    dist.print0(f'[W&B] log failed: {_e}')
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Built-in validation hook (runs on schedule; blocks training).
        try:
            cur_kimg = cur_nimg // 1000
            maybe_validate(
                step_tick=cur_tick,
                step_kimg=int(cur_kimg),
                net_ema=ema,
                run_dir=run_dir,
                dataset_kwargs=dataset_kwargs,
                validation_kwargs=validation_kwargs,
                wandb_run=wandb_run,
            )
        except Exception as _e:
            dist.print0(f'[VAL] validation failed: {_e}')

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
    # Close W&B run.
    try:
        if wandb_run is not None and dist.get_rank() == 0:
            import wandb as _wandb
            _wandb.finish()
    except Exception:
        pass

#----------------------------------------------------------------------------
