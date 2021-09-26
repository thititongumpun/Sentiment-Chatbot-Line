import re

deliverlyServicePatten = "^เรียกรถ|รถ|(?:.*รถเข้ารับพัสดุ)|พัศดุ|.*ราคารถ|.*เช็ค|.*พัสดุ|.*สินค้า|.*สั่งของ|.*จัดส่ง|.*กี่วัน|.*รับ|.*เคลม|.*ของ|.*ส่ง|.*แตก|.*ยัง"
servicePattern = ".*บริการ|บริการ|บริการจากเรา^|^ตรวจสอบค่าบริการ$|^.*ราคา$|^ราคา|ราคา$|.*พนักงาน|.*สอบถาม|.*ผู้ติดต่อ|.*ตรวจสอบค่าบริการ|.*ติดต่อ|.*โทรไม่|.*แย่|.*ห่วย|.*ถาม|.*ตาม \
                  .*|กี่บาท|.*สาขา|(?:.*สมัคร)|.*ที่อยู่"
promotionPattern = ".*โปรโมชั่น|โปรโมชัน|^รายละเอียดโปรโมชั่น$|^โปรโมชั่น$"

def service_type(sentimentText) -> str:
  if re.findall(promotionPattern,sentimentText) or re.match(promotionPattern, sentimentText):
    return "ด้านโปรโมชั่น"
  elif re.findall(servicePattern,sentimentText) or re.match(servicePattern, sentimentText):
    return "ด้านบริการ"
  elif re.findall(deliverlyServicePatten,sentimentText) or re.match(deliverlyServicePatten, sentimentText):
    return "บริการด้านขนส่ง"
  else:
    return "อื่นๆ"

