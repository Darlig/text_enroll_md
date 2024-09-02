#__all__ = ['DoubleAE']
# new model append in m_dict
# from model import DoubleAE
# from model import CNNASR
# from model import THUEnergy
# from model import CNNASR_TEMB
# from model import CNN_GRAVE
# from model import LocationAtt
# from model import CTCModel
# from model import TransformerASR
# from model import TransformerKWS
# from model import CNN_GRAVEmlp
# from model import SpeechExtractor
# from model import LocationAttPIT
# from model import LocationAttCross
# from model import TransformerKWSPIT
# from model import LocationAttCFN
# from model import MobileNetSY
# from model import MobileNetSYL2
# from model import LocationAttCN
# from model import LocationAttCNL2
# from model import MobileNetSYMixUp
# from model import MobileNetSYL2FullHomo
# from model import MobileNetSYL2NoNeg
# from model import LocationAttCNFullHomo
# from model import HomoSE
# from model import CNNGSC
# from model import HomoGSC
# from model import HomoGSC_Clean
# from model import MobileNetSYL2FullHomo
# from model import MobileNetSYL2
# from model import HomoASR
# from model import CTCASR
# from model import HomoASRTDNN
# from model import ASRTDNN
# from model import HomoASRCTDNN
# from model import MixUp
# from model import HomoGSCMixUp
# from model import KWSCNN
# from model import EfficientNet
# from model import BCResNet
# from model import EfficientNetWeight
# from model import TransformerVSR
from model import ConformerHintSSVocoder
from model import AEDKWSASR
from model import AEDKWSASRPhone
from model import AEDASR
from model import AEDKWSASRPhoneUnet

m_dict = {
    'ConformerHintSSVocoder': ConformerHintSSVocoder.ConformerHintSSVocoder,
    'AEDKWSASR': AEDKWSASR.AEDKWSASR,
    'AEDKWSASRPhone': AEDKWSASRPhone.AEDKWSASRPhone,
    'AEDASR': AEDASR.AEDASR,
    'AEDKWSASRPhoneUnet': AEDKWSASRPhoneUnet.AEDKWSASRPhoneUnet
}
