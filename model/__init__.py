#__all__ = ['DoubleAE']
# new model append in m_dict
from model import AEDKWSASR
from model import AEDKWSASRPhone
from model import AEDASR
from model import AEDKWSASRPhoneUnet
from model import TransformerKWSPhone
from model import TransformerKWSPhone_noaliloss
from model import TransformerKWSPhone_eval_steps
from model import TransformerKWSPhone_nocross

m_dict = {
    'AEDKWSASR': AEDKWSASR.AEDKWSASR,
    'AEDKWSASRPhone': AEDKWSASRPhone.AEDKWSASRPhone,
    'AEDASR': AEDASR.AEDASR,
    'AEDKWSASRPhoneUnet': AEDKWSASRPhoneUnet.AEDKWSASRPhoneUnet,
    'TransformerKWSPhone': TransformerKWSPhone.TransformerKWSPhone,
    'TransformerKWSPhone_noaliloss': TransformerKWSPhone_noaliloss.TransformerKWSPhone_noaliloss,
    'TransformerKWSPhone_eval_steps': TransformerKWSPhone_eval_steps.TransformerKWSPhone_eval_steps,
    'TransformerKWSPhone_nocross': TransformerKWSPhone_nocross.TransformerKWSPhone_nocross
}
