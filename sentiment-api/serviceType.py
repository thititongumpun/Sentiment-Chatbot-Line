import re

parcelPattern = "^พัสดุ|พัสดุ|พัสดุ$"
deliverlyServicePatten = "^เรียกรถ|รถ|รถเข้ารับพัสดุ$"
servicePattern = "^บริการ|บริการ|บริการจากเรา^|^ตรวจสอบค่าบริการ$|^.*ราคา$|^ราคา|ราคา$"
promotionPattern = "^โปรโมชั่น|โปรโมชั่น|^รายละเอียดโปรโมชั่น$|^โปรโมชั่น$"

def service_type(sentimentText) -> str:
  if re.findall(parcelPattern,sentimentText) or re.match(parcelPattern, sentimentText):
    return "บริการด้านพัสดุ"
  elif re.findall(deliverlyServicePatten,sentimentText) or re.match(deliverlyServicePatten, sentimentText):
    return "บริการด้านเรียกรถ"
  elif re.findall(servicePattern,sentimentText) or re.match(servicePattern, sentimentText):
    return "ด้านบริการ"
  elif re.findall(promotionPattern,sentimentText) or re.match(promotionPattern, sentimentText):
    return "ด้านโปรโมชั่น"
  else:
    return "อื่นๆ"

