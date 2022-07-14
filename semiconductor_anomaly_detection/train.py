from codes.model import Anomaly_Detection_model
from codes.utils import preprocess

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

    ad_model.fit("xgb", x, y, 3, xgb_config)
