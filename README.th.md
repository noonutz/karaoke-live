# karaoke-live

[English](README.md) · **ไทย**

ตัดเสียงร้องนำออกจากอะไรก็ตามที่คอมพิวเตอร์กำลังเล่นอยู่ แบบสด ๆ แล้วร้องทับได้
ทำมาสำหรับงานที่เครื่องมือทั่วไปไม่ตอบ: ไม่ใช่ "ฟังเสียงพูดในหนังให้ชัด" แต่เป็น
**"เก็บดนตรีไว้ ตัดคนร้องออก"**

สามเอนจิน ใช้ interface เดียวกัน:

| engine | ทำอะไร | latency | ต้องมีอะไร |
|---|---|---|---|
| `center` (ค่าเริ่มต้น) | ตัดเสียงกลางภาพด้วยการกลับเฟส, คืนเบสให้, ปรับได้ 0-100% | **0** | numpy + ffmpeg |
| `channel` | เลือกข้างเดียว (แหล่ง multiplex: ร้องชัดข้างซ้าย ดนตรีข้างขวา) | **0** | numpy + ffmpeg |
| `demucs` | แยก 2 stem ด้วยโมเดลจริง แบบแบ่ง chunk | **~chunk_s** (ตั้งต้น 2.5 วิ) | + torch, demucs |

`center` คือตัวที่ใช้ได้เสมอ: ไม่ต้องโหลดโมเดล ไม่ต้องมี GPU ไม่เพิ่ม latency
และเป็นเทคนิคเดียวกับที่แอมป์คาราโอเกะใช้มาตั้งแต่ยุค 90 — ต่างกันแค่ใช้ครอสโอเวอร์
อันดับ 4 เพื่อไม่ให้เบสหาย และชดเชยระดับเสียงตอนหมุนปุ่ม เพื่อให้หมุนแรงขึ้นแล้ว
ดนตรีไม่เบาลง

`demucs` เสียงดีกว่า (คง stereo, กลองอยู่ครบ, ไม่มีเสียงร้องเงากลับเฟส)
แลกกับ latency หนึ่ง chunk — เป็นการแลกที่ตรงไปตรงมา: โมเดลแบบ non-causal
จะปล่อยเสียงออกไม่ได้จนกว่าจะได้ยินเสียงที่อยู่ถัดไปก่อน

## ติดตั้ง

```bash
git clone https://github.com/noonutz/karaoke-live && cd karaoke-live
python3 -m venv .venv && .venv/bin/pip install -e .        # center/channel เท่านั้น
.venv/bin/pip install -e '.[live,ml]'                     # + เสียงจริง, + demucs
```

macOS ต้อง `brew install ffmpeg portaudio` เพิ่ม; Windows ต้องมี ffmpeg ใน PATH

## ใช้งาน

```bash
# มีอุปกรณ์เสียงอะไรบ้าง
karaokelive devices

# เสียงระบบ -> หูฟัง ตัดร้อง 70%
karaokelive live --capture "BlackHole 2ch" --output "MacBook Pro Speakers" --vocal-level 70

# คุณภาพดีกว่า หน่วง ~2.5 วิ
karaokelive live --capture "BlackHole 2ch" --output "MacBook Pro Speakers" \
    --engine demucs --vocal-level 100

# ออฟไลน์: เรนเดอร์เป็นไฟล์ instrumental
karaokelive file song.mp3 song_karaoke.wav --vocal-level 100 --no-mono-guard

# ออฟไลน์ คุณภาพดีสุด: แยก stem จริงผ่าน demucs CLI
karaokelive prerender song.mp3 stems/
```

กด Ctrl-C เพื่อหยุด

## การต่อสาย (จุดที่คนพลาดมากที่สุด)

**macOS:** ติดตั้ง [BlackHole](https://github.com/ExistentialAudio/BlackHole) 2ch
(`brew install blackhole-2ch`) แล้วตั้ง system output ไปที่มัน จากนั้นให้
`--capture` ชี้ไปที่ตัวนั้น เบราว์เซอร์จะเงียบ เพราะทุกเสียงไหลผ่าน karaoke-live
แล้ว ถ้าอยากส่งเข้า OBS พร้อมกัน ให้สร้าง **Multi-Output Device** ใน Audio MIDI
Setup (BlackHole + อินเทอร์เฟซ) หรือให้ OBS ดึงจากอินเทอร์เฟซแทน

**Windows:** ติดตั้ง [VB-CABLE](https://vb-audio.com/Cable/) ตั้ง *CABLE Input*
เป็นอุปกรณ์เล่นเสียงเริ่มต้น แล้วใช้ `--capture "CABLE Output"` ถ้าต้องการหลายสาย
ให้เพิ่ม VB-CABLE A+B แล้วใช้ `--capture "CABLE-A Output"`

**OBS:** เพิ่ม *Audio Output Capture* ที่สายเสมือนที่เล่นออกไป หรือใช้
*Audio Input Capture* ถ้าส่งเสียงที่ประมวลผลแล้วกลับผ่านอินเทอร์เฟซ เพราะทั้งเส้น
ทางดนตรีถูกหน่วงเท่ากันหมด นักร้องกับดนตรีจึงไม่หลุดกัน — สิ่งที่เหลื่อมคือ
**ภาพ** ซึ่งไม่รู้เรื่องความหน่วงเลย ปิดภาพซะ หรือใส่ delay ใน OBS เท่ากัน

## เรื่องที่ควรรู้ก่อนเชื่อมัน

- **กับดักคือ mono downmix** — `center` ที่ 100% จะทำให้ L กับ R กลับเฟสกัน พอรวมเป็น
  mono แล้วดนตรีหายเกือบหมด คนฟังผ่านลำโพงมือถือหรือ Bluetooth ตัวเดียวจะแทบไม่ได้ยิน
  อะไร ค่าเริ่มต้นจึงจำกัดไว้ที่ 85% (`--no-mono-guard` ถ้าอยากปลดล็อก) — ลำโพงหรือ
  หูฟังปกติไม่เป็นปัญหา
- **เสียงกลางภาพถูกตัดไปด้วย** — กระเดื่อง สแนร์ และเบส อยู่กลางเหมือนเสียงร้อง
  จึงมีครอสโอเวอร์ (`--bass-hz` ค่าเริ่มต้น 140 Hz) ที่เติมทุกอย่างใต้จุดนั้นกลับมา
- **แหล่ง multiplex คือของฟรี** — ถ้าเสียงร้องอยู่ข้างเดียว ใช้ `--engine channel`
  แล้วข้าม DSP ไปเลย
- **คนร้องได้ยินเสียงเงามากกว่าคนฟัง** — ในหูฟัง เสียงร้องที่เหลือจะชัดมากสำหรับคุณ
  แต่คนฟังผ่านลำโพงจะไม่ได้ยินเลย ถ้ารำคาญ นั่นคือเหตุผลที่ `demucs` มีอยู่
- **license ของ weights ไม่ใช่ license ของโค้ด** — โค้ด Demucs เป็น MIT แต่ weights
  ที่เทรนมาไม่อยู่ใต้ MIT และข้อมูลเทรนบางส่วนก็ไม่มีเอกสารชัดเจน ใช้เองไม่มีปัญหา
  แต่ถ้าจะแจกจ่ายต้องคิดหนัก repo นี้ไม่เคย commit weights — `demucs` โหลดเอง
  ตอนติดตั้ง

## สถานะ

- `center` และ `channel`: มีเทสต์ครบ ทดสอบกับไฟล์จริงแบบ end-to-end แล้ว
  (โทน 440 Hz กลางภาพลดลง 33.8 dB ขณะที่เสียงแพนข้างขยับไม่ถึง 0.1 dB)
- `demucs`: การแบ่ง chunk, crossfade และการคำนวณ latency มีเทสต์ด้วยตัวแทน model
  (ผ่าน signal แบบ bit-exact, ตรวจ alignment ด้วยการค้นหา lag เอง) แต่ยังไม่เคยรัน
  โมเดลจริงบนเครื่องที่พัฒนา — ไม่มี torch ที่นั่น ถือเป็นเส้นทางทดลอง
- โหมด live ต้องมีอุปกรณ์เสียงจริง พัฒนาบนเครื่อง headless จึงเป็นส่วนที่ถูกทดสอบ
  น้อยที่สุด เริ่มจาก `devices` แล้วรัน `--seconds 10` คือการทดสอบที่สมเหตุสมผล

## ที่มาของงาน

ยืนบนไหล่ของ [Poise Voice Isolator](https://github.com/chabandou/poise-voice-isolator)
(MIT) ในแง่รูปร่างของปัญหา — มันเสียบโมเดลเข้ากับ audio graph ของระบบ และแก้
ทิศทางตรงกันข้าม (เก็บเสียงพูด ทิ้งที่เหลือ) ส่วน
[Demucs](https://github.com/facebookresearch/demucs) รับงานหนักในเอนจิน ML
และ [BlackHole](https://github.com/ExistentialAudio/BlackHole) /
[VB-CABLE](https://vb-audio.com/Cable/) จัดการเรื่องการต่อสาย

สัญญาอนุญาต MIT
