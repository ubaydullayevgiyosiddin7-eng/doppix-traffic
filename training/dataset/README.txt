Do'ppiX detektor dataseti — doppix_det_20260926_134722
Yaratilgan: 2026-09-26 13:47
Format: YOLO · faqat TASDIQLANGAN kadrlar
Split: har videoning oxirgi 10% i val

train: 192 rasm, 11604 box
val:   22 rasm, 1386 box

Klass            train    val
0: person          5692    636
1: bicycle           13      0
2: car             4901    637
3: motorcycle        41      0
4: bus              403     41
5: truck            188     28
6: light_red        219     20
7: light_not_red    147     24

Videolar (train / val):
  C3896.MP4: 122 / 14
  C3905.MP4: 70 / 8

O'qitish (ZIP ochilgan papkada):
  yolo detect train model=yolo26m.pt data=data.yaml imgsz=1280 epochs=50 seed=42 deterministic=True
