from codes.model import Anomaly_Detection_model
from codes.utills import preprocess

if __name__ == '__main__':
    # train 데이터 전처리 후 불러오기
    x, y = preprocess()

    # 모델 할당
    ad_model = Anomaly_Detection_model()

    # xgb 하이퍼 파리미터
    xgb_config = {
        'height':1, 
        'grid' :True, 
        'importance_type':'gain', 
        'show_values':False, 
        'max_num_features':20
    }

    # lda 하이퍼 파리미터 
    lda_config = {
        'solver':'svd'
    }
    
    # 각 모델 학습
    ad_model.fit("xgb", x, y, 3, xgb_config)
    ad_model.fit("lda", x, y, 3, lda_config)
