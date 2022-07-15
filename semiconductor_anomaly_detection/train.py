from codes.model import Anomaly_Detection_model
from codes.utills import preprocess

if __name__ == '__main__':
    x, y = preprocess()

    ad_model = Anomaly_Detection_model()

    xgb_config = {
        'height':1, 
        'grid' :True, 
        'importance_type':'gain', 
        'show_values':False, 
        'max_num_features':20
    }

    lda_config = {
        'solver':'svd'
    }
    
    ad_model.fit("xgb", x, y, 3, xgb_config)
    ad_model.fit("lda", x, y, 3, xgb_config)
