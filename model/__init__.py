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
from model import TransformerKWSPhone_nocross_w_ctc
from model import TransformerKWSPhone_nocross_w_ctc_tmp_debug2_equal_to_no_spec_mask
from model import TransformerKWSPhone_nocross_w_ctc_tmp_debug1_target_mask
from model import TransformerKWSPhone_nocross_wo_ctc
from model import TransformerKWSPhone_nocross_w_ctc_kw_adapter
from model import TransformerKWSPhone_nocross_w_ctc_kw_adapter_nonlinear

m_dict = {
    'AEDKWSASR': AEDKWSASR.AEDKWSASR,
    'AEDKWSASRPhone': AEDKWSASRPhone.AEDKWSASRPhone,
    'AEDASR': AEDASR.AEDASR,
    'AEDKWSASRPhoneUnet': AEDKWSASRPhoneUnet.AEDKWSASRPhoneUnet,
    'TransformerKWSPhone': TransformerKWSPhone.TransformerKWSPhone,
    'TransformerKWSPhone_noaliloss': TransformerKWSPhone_noaliloss.TransformerKWSPhone_noaliloss,
    'TransformerKWSPhone_eval_steps': TransformerKWSPhone_eval_steps.TransformerKWSPhone_eval_steps,
    'TransformerKWSPhone_nocross': TransformerKWSPhone_nocross.TransformerKWSPhone_nocross,
    'TransformerKWSPhone_nocross_w_ctc': TransformerKWSPhone_nocross_w_ctc.TransformerKWSPhone_nocross_w_ctc,
    'TransformerKWSPhone_nocross_w_ctc_tmp_debug2_equal_to_no_spec_mask': TransformerKWSPhone_nocross_w_ctc_tmp_debug2_equal_to_no_spec_mask.TransformerKWSPhone_nocross_w_ctc_tmp_debug2_equal_to_no_spec_mask,
    'TransformerKWSPhone_nocross_w_ctc_tmp_debug1_target_mask': TransformerKWSPhone_nocross_w_ctc_tmp_debug1_target_mask.TransformerKWSPhone_nocross_w_ctc_tmp_debug1_target_mask,
    'TransformerKWSPhone_nocross_wo_ctc': TransformerKWSPhone_nocross_wo_ctc.TransformerKWSPhone_nocross_wo_ctc,
    'TransformerKWSPhone_nocross_w_ctc_kw_adapter': TransformerKWSPhone_nocross_w_ctc_kw_adapter.TransformerKWSPhone_nocross_w_ctc_kw_adapter,
    'TransformerKWSPhone_nocross_w_ctc_kw_adapter_nonlinear': TransformerKWSPhone_nocross_w_ctc_kw_adapter_nonlinear.TransformerKWSPhone_nocross_w_ctc_kw_adapter_nonlinear
}
