model_configs = {
    # "DNN": {
    #     "class_name": "DNNModel",
    #     "meta": {
    #         "include_glove": False
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # }, 
    # "DNN_Glove": {
    #     "class_name": "DNNModel",
    #     "meta": {
    #         "include_glove": True
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # },
    # "SimpleRNN": {
    #     "class_name": "SimpleRNNModel",
    #     "meta": {
    #         "include_glove": False
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # }, 
    # "SimpleRNN_Glove": {
    #     "class_name": "SimpleRNNModel",
    #     "meta": {
    #         "include_glove": True
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # },
    # "CNN": {
    #     "class_name": "CNNModel",
    #     "meta": {
    #         "include_glove": False
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # }, 
    # "CNN_Glove": {
    #     "class_name": "CNNModel",
    #     "meta": {
    #         "include_glove": True
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # },
    # "LSTM": {
    #     "class_name": "LSTMModel",
    #     "meta": {
    #         "include_glove": False
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # }, 
    # "LSTM_Glove": {
    #     "class_name": "LSTMModel",
    #     "meta": {
    #         "include_glove": True
    #     }, 
    #     "params": {
    #         "dropout": 0.3,
    #         "initial_learning_rate": 0.01
    #     }
    # },
    "StackedLSTM": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
        }
    }, 
    "StackedLSTM_Glove": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
        }
    },
   "StackedBiLSTM": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
        }
    }, 
    "StackedBiLSTM_Glove": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
        }
    },
    "BiLSTM": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
        }
    }, 
    "BiLSTM_Glove": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
        }
    },
    "HybridModel": {
        "class_name": "HybridModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
        }
    },
    "HybridModel_Glove": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
        }
    },
}