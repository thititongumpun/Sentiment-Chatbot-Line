import re

parcelPattern = ".*พัสดุ|.*สินค้า|.*สั่งของ|.*จัดส่ง|.*กี่วัน|.*รับ|.*เคลม"
deliverlyServicePatten = "^เรียกรถ|รถ|(?:.*รถเข้ารับพัสดุ)|.*ราคารถ"
servicePattern = ".*บริการ|บริการ|บริการจากเรา^|^ตรวจสอบค่าบริการ$|^.*ราคา$|^ราคา|ราคา$|.*พนักงาน|.*สอบถาม|.*ผู้ติดต่อ|.*ตรวจสอบค่าบริการ|.*ติดต่อ|.*โทรไม่|.*แย่|.*ห่วยแตก|.*ถาม|.*ตาม"
promotionPattern = ".*โปรโมชั่น|โปรโมชัน|^รายละเอียดโปรโมชั่น$|^โปรโมชั่น$"

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

