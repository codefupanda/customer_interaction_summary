model_configs = {
    "DNN": {
        "class_name": "DNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "DNN_Glove": {
        "class_name": "DNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "SimpleRNN": {
        "class_name": "SimpleRNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "SimpleRNN_Glove": {
        "class_name": "SimpleRNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "CNN": {
        "class_name": "CNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "CNN_Glove": {
        "class_name": "CNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "LSTM": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "LSTM_Glove": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "StackedLSTM": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "StackedLSTM_Glove": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
   "StackedBiLSTM": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "StackedBiLSTM_Glove": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "BiLSTM": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    }, 
    "BiLSTM_Glove": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "HybridModel": {
        "class_name": "HybridModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
    "HybridModel_Glove": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "dropout": 0.3,
        }
    },
}