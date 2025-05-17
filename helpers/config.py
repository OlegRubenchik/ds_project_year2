import numpy as np

PRICE_REG_CONFIG = [
    'build_area',
    'construction_year',
    'number_of_bedrooms',
    'floor_number',
    'latitude',
    'longitude',
    'loccell',
]

MODEL_CONFIG = {
    'transformer': {
        'enabled': True,  # Whether to use transformation
        'class': 'TransformedRegressor',
        'params': {
            'transform_fn': np.log,
            'inverse_fn': np.exp
        }
    },
    'base_model': {
        'params': {
            'random_state': 52,
            'verbose': 1,
            'max_depth': 15,
            'n_jobs': -1  # Use all cores for the RandomForest itself
        }
    },
    'grid_search': {
        'params': {
            'n_estimators': [100, 300, 500, 700, 1000, 2000],
            'max_depth': [10, 15]
        },
        'cv': 3,
        'n_jobs': -1
    }
}
    