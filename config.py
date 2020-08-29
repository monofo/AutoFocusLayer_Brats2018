import torch


class TrainGlobalConfig:

    base_dir = "ex5"

    id = 'AFN1' # Choose from the following options: AFN1-6, Basic, ASPP_c, ASPP_s.
    padding_list = [0,4,8,12] # list of the paddings in the parallel convolutions
    dilation_list = [2,6,10,14]
    num_branches = 4

    crop_size = [75,75,75]
    center_size = [47, 47, 47]
    num_classes = 4
    num_input = 5
    num_workers = 4

    batch_size = 10
    n_epochs = 300
    lr = 1e-3


    root_path = "~/dataset/BRATS2018/Train"
    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.8,
        patience=5,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------
