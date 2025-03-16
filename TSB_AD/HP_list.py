
Multi_algo_HP_dict = {
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200],
        'max_features': [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'metric': ['minkowski', 'manhattan', 'euclidean']
    },    
    'PCA': {
        'n_components': [0.25, 0.5, 0.75, None]
    },        
    'HBOS': {
        'n_bins': [5, 10, 20, 30, 40],
        'tol': [0.1, 0.3, 0.5, 0.7]
    },
    'OCSVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'nu': [0.1, 0.3, 0.5, 0.7]
    },        
    'MCD': {
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'KNN': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'method': ['largest', 'mean', 'median']
    },        
    'KMeansAD': {
        'n_clusters': [10, 20, 30, 40],
        'window_size': [10, 20, 30, 40]
    },
    'COPOD': {
        'HP': [None]
    },    
    'CBLOF': {
        'n_clusters': [4, 8, 16, 32],
        'alpha': [0.6, 0.7, 0.8, 0.9]
    },
    'EIF': {
        'n_trees': [25, 50, 100, 200]
    },   
    'RobustPCA': {
        'max_iter': [500, 1000, 1500]
    },
    'AutoEncoder': {
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },  
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },    
    'OFA': {
        'win_size': [50, 100, 150]
    } 
}


Optimal_Multi_algo_HP_dict = {
    'IForest': {'n_estimators': 25, 'max_features': 0.8},
    'LOF': {'n_neighbors': 50, 'metric': 'euclidean'},    
    'PCA': {'n_components': 0.25},        
    'HBOS': {'n_bins': 30, 'tol': 0.5},
    'OCSVM': {'kernel': 'rbf', 'nu': 0.1},        
    'MCD': {'support_fraction': 0.8},
    'KNN': {'n_neighbors': 50, 'method': 'mean'},        
    'KMeansAD': {'n_clusters': 10, 'window_size': 40},
    'KShapeAD': {'n_clusters': 20, 'window_size': 40},
    'COPOD': {'n_jobs':1},    
    'CBLOF': {'n_clusters': 4, 'alpha': 0.6},
    'EIF': {'n_trees': 50},   
    'RobustPCA': {'max_iter': 1000},
    'AutoEncoder': {'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 150, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.001},  
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'OmniAnomaly': {'win_size': 100, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},  
    'Donut': {'win_size': 60, 'lr': 0.001},  
    'TimesNet': {'win_size': 96, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.001},
    'OFA': {'win_size': 50}
}


Uni_algo_HP_dict = {
    'Sub_IForest': {
        'periodicity': [1, 2, 3],
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'Sub_LOF': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'POLY': {
        'periodicity': [1, 2, 3],
        'power': [1, 2, 3, 4]
    },
    'MatrixProfile': {
        'periodicity': [1, 2, 3]
    },
    'NORMA': {
        'periodicity': [1, 2, 3],
        'clustering': ['hierarchical', 'kshape']
    },
    'SAND': {
        'periodicity': [1, 2, 3]
    }, 
    'Series2Graph': {
        'periodicity': [1, 2, 3]
    },
    'Sub_PCA': {
        'periodicity': [1, 2, 3],
        'n_components': [0.25, 0.5, 0.75, None]
    },
    'Sub_HBOS': {
        'periodicity': [1, 2, 3],
        'n_bins': [5, 10, 20, 30, 40]
    },
    'Sub_OCSVM': {
        'periodicity': [1, 2, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    'Sub_MCD': {
        'periodicity': [1, 2, 3],
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'Sub_KNN': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50],
    },
    'KMeansAD_U': {
        'periodicity': [1, 2, 3],
        'n_clusters': [10, 20, 30, 40],
    },
    'KShapeAD': {
        'periodicity': [1, 2, 3]
    },
    'AutoEncoder': {
        'window_size': [50, 100, 150],
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'OFA': {
        'win_size': [50, 100, 150]
    },    
    'Lag_Llama': {
        'win_size': [32, 64, 96]
    },    
    'Chronos': {
        'win_size': [50, 100, 150]
    },
    'TimesFM': {
        'win_size': [32, 64, 96]
    },
    'MOMENT_ZS': {
        'win_size': [64, 128, 256]
    },
    'MOMENT_FT': {
        'win_size': [64, 128, 256]
    }
}

Optimal_Uni_algo_HP_dict = {
    'Sub_IForest': {'periodicity': 1, 'n_estimators': 150},
    'IForest': {'n_estimators': 200},
    'Sub_LOF': {'periodicity': 2, 'n_neighbors': 30},
    'LOF': {'n_neighbors': 50},
    'POLY': {'periodicity': 1, 'power': 4},
    'MatrixProfile': {'periodicity': 1},
    'NORMA': {'periodicity': 1, 'clustering': 'kshape'},
    'SAND': {'periodicity': 1},
    'Series2Graph': {'periodicity': 1},
    'SR': {'periodicity': 1},
    'Sub_PCA': {'periodicity': 1, 'n_components': None},        
    'Sub_HBOS': {'periodicity': 1, 'n_bins': 10},
    'Sub_OCSVM': {'periodicity': 2, 'kernel': 'rbf'},        
    'Sub_MCD': {'periodicity': 3, 'support_fraction': None},
    'Sub_KNN': {'periodicity': 2, 'n_neighbors': 50}, 
    'KMeansAD_U': {'periodicity': 2, 'n_clusters': 10},
    'KShapeAD': {'periodicity': 1},
    'FFT': {},
    'Left_STAMPi': {},
    'AutoEncoder': {'window_size': 100, 'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 100, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.0001},
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'OmniAnomaly': {'win_size': 5, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},
    'Donut': {'win_size': 60, 'lr': 0.0001},  
    'TimesNet': {'win_size': 32, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.0001},
    'OFA': {'win_size': 50},
    'Lag_Llama': {'win_size': 96},
    'Chronos': {'win_size': 100},
    'TimesFM': {'win_size': 96},
    'MOMENT_ZS': {'win_size': 64},
    'MOMENT_FT': {'win_size': 64},
    'M2N2': {}
}