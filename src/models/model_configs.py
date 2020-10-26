model_configs = {
    "DNN": {
        "class_name": "DNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "DNN_Glove": {
        "class_name": "DNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "SimpleRNN": {
        "class_name": "SimpleRNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "SimpleRNN_Glove": {
        "class_name": "SimpleRNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "CNN": {
        "class_name": "CNNModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "CNN_Glove": {
        "class_name": "CNNModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "LSTM": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "LSTM_Glove": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "StackedLSTM": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "StackedLSTM_Glove": {
        "class_name": "StackedLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
   "StackedBiLSTM": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "StackedBiLSTM_Glove": {
        "class_name": "StackedBiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "BiLSTM": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    }, 
    "BiLSTM_Glove": {
        "class_name": "BiLSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "HybridModel": {
        "class_name": "HybridModel",
        "meta": {
            "include_glove": False
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
    "HybridModel_Glove": {
        "class_name": "LSTMModel",
        "meta": {
            "include_glove": True
        }, 
        "params": {
            "recurrent_dropout": 0
        }
    },
}