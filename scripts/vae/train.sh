#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATE_TIME=$(date +"%Y-%m-%d_%H-%M")

taskset -c 0-47 python src/vae/train.py \
    model.loss.use_triplet=true \
    experiment_name=triplet_loss \
    training.learning_rate=0.0001 \
    optimization.optimizer.weight_decay=0.0001 \
    model.hidden_dims=[512,256,128] \
    model.beta_scheduler.beta_end=0.001 \
    model.beta_scheduler.warmup_epochs=25 \
    model.loss.free_bits=0.1 \
    optimization.scheduler.patience=25 \
    model.loss.triplet_weight=1.0 \
    model.loss.triplet_margin=0.5 \
    model.loss.triplet_mining_strategy=batch_all \
    dataset.normalization.enabled=true \
    dataset.normalization.method=ddsp \
    dataset.normalization.save_normalizer=true \
    training.early_stopping.enabled=false \
    dataset.batch_size=1024 \
    training.dataloader_num_workers=16 \
    training.prefetch_factor=4 \
    training.persistent_workers=true \
    training.pin_memory=true \
    dataset.cache_features=true \
    dataset.eager_cache=true    

    # data.valid_max_samples=5000
    # data.train_max_samples=5000 \
    # data.valid_max_samples=800 