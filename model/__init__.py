#__all__ = ['DoubleAE']
#from model import AEDKWSASRPhone
#from model import SequentialASR
#from model import AEDASR
#from model import PITSOT
#from model import ConformerCTC
#from model import EncoderDecoder
#from model import BCResNet
#from model import CohortChainASR
#from model import PITSOTKWS
from model import AEDKWSASRPhoneUnet
#from model import CohortChainASRTripleLoss
#from model import CohortASR
from model import TransformerKWSPhoneUnet

m_dict = {
    #'AEDKWSASRPhone': AEDKWSASRPhone.AEDKWSASRPhone,
    #'SequentailASR': SequentialASR.SequentailASR,
    #'AEDASR': AEDASR.AEDASR,
    #'PITSOT': PITSOT.PITSOT,
    #'ConformerCTC': ConformerCTC.ConformerCTC,
    #'EncoderDecoder': EncoderDecoder.EncoderDecoder,
    #'AEDKWSASRPhoneIP': AEDKWSASRPhoneIP.AEDKWSASRPhoneIP,
    #'BCResNet': BCResNet.BCResNet,
    #'CohortChainASR': CohortChainASR.CohortChainASR,
    #'PITSOTKWS': PITSOTKWS.PITSOTKWS,
    'AEDKWSASRPhoneUnet': AEDKWSASRPhoneUnet.AEDKWSASRPhoneUnet,
    #'CohortChainASRTripleLoss': CohortChainASRTripleLoss.CohortChainASRTripleLoss,
    #'CohortASR': CohortASR.CohortASR,
    'TransformerKWSPhoneUnet': TransformerKWSPhoneUnet.TransformerKWSPhoneUnet
}
