tpiv-simulations/
│
├── src/
│   ├── models/
│   │   ├── net.py                 # Net class (teacher/student)
│   │   └── __init__.py
│   │
│   ├── training/
│   │   ├── train_student.py        # train_student_on_data
│   │   ├── mse.py                  # S_MSE and related metrics
│   │   └── __init__.py
│   │
│   ├── data/
│   │   ├── generate_data.py        # x_train, y_train generation
│   │   └── __init__.py
│   │
│   ├── experiment/
│   │   ├── run_experiment.py       # your full experiment loop
│   │   ├── save_utils.py           # CSV, pickle, directories
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── seed.py                 # init_torch
│   │   ├── config.py               # convert_numeric_config
│   │   ├── logging_utils.py        # log formatting, header/footer
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── experiments/
│   └── ERM/ERM_S/
│       ├── ERM_S_exp_cluster.py    # becomes much smaller: CLI wrapper
│       └── ...
│
└── results/