from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'app' / 'uploads'
    MODELS_FOLDER = BASE_DIR / 'models'
    
    # Allowed extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
    
    # Image settings
    IMG_SIZE = 640
    
    # Preprocessing methods
    PREPROCESSING_METHODS = {
        'gaussian_clahe': 'Gaussian + CLAHE',
        'otsu_clahe_nlm': 'Otsu + CLAHE + NLM',
        'test': 'NLM + CLAHE (Optimized)',
        'test2': 'Full Pipeline (Crop + Scanlines + Otsu + CLAHE + NLM)',
        'unprocessed': 'Unprocessed (Baseline)'
    }
    
    # Views
    VIEWS = {
        'cc': 'Craniocaudal (CC)',
        'mlo': 'Mediolateral Oblique (MLO)',
        'cc_mlo': 'Combined (CC + MLO)'
    }
    
    # Model performance data 
    MODEL_PERFORMANCE = {
        'cc': {
            'gaussian_clahe': {'precision': 0.39147, 'recall': 0.41971, 'mAP50': 0.31389},
            'otsu_clahe_nlm': {'precision': 0.43046, 'recall': 0.36178, 'mAP50': 0.33512},
            'test': {'precision': 0.51706, 'recall': 0.42308, 'mAP50': 0.37259},
            'test2': {'precision': 0.28151, 'recall': 0.38714, 'mAP50': 0.27414},
            'unprocessed': {'precision': 0.41328, 'recall': 0.32051, 'mAP50': 0.32051}
        },
        'mlo': {
            'gaussian_clahe': {'precision': 0.3369, 'recall': 0.46473, 'mAP50': 0.337},
            'otsu_clahe_nlm': {'precision': 0.34856, 'recall': 0.27244, 'mAP50': 0.22722},
            'test': {'precision': 0.33962, 'recall': 0.26923, 'mAP50': 0.24787},
            'test2': {'precision': 0.41732, 'recall': 0.5, 'mAP50': 0.40305},
            'unprocessed': {'precision': 0.41165, 'recall': 0.45373, 'mAP50': 0.40261}
        },
        'cc_mlo': {
            'gaussian_clahe': {'precision': 0.55083, 'recall': 0.34695, 'mAP50': 0.41869},
            'otsu_clahe_nlm': {'precision': 0.50167, 'recall': 0.36866, 'mAP50': 0.39236},
            'test': {'precision': 0.47497, 'recall': 0.48063, 'mAP50': 0.44842},
            'test2': {'precision': 0.55571, 'recall': 0.38814, 'mAP50': 0.41717},
            'unprocessed': {'precision': 0.45874, 'recall': 0.42236, 'mAP50': 0.3862}
        }
    }