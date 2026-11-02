// =====================================================================
//  Sim Pedal + Handbrake  (RP2040-Zero)  単一ファイル版
//  Logicool G29 ペダル(DE-9) 変換 + 自作サイドブレーキ を1つのUSBジョイスティックに統合
//
//  Arduino IDE 設定:
//    ボード      : Waveshare RP2040 Zero (Earle Philhower core)
//    USB Stack   : Adafruit TinyUSB   ← 必須
//  外部ライブラリ不要 (Adafruit TinyUSB のみ)。
//
//  配線 (※RP2040のADCは最大3.3V。ペダルへの電源は必ず 3V3 から):
//    DE-9 pin1          -> GND
//    DE-9 pin2 (Gas)    -> GP27
//    DE-9 pin3 (Brake)  -> GP28
//    DE-9 pin4 (Clutch) -> GP29
//    DE-9 pin6 + pin9   -> 3V3  (pin6は接続検出も兼ねる)
//    サイドブレーキ(3P) : 信号 -> GP26, 3V3, GND
//
//  HID軸: X=アクセル / Y=ブレーキ / Z=クラッチ / Rx=サイドブレーキ
// =====================================================================
#include "Adafruit_TinyUSB.h"

// ---------------------------------------------------------------------
// ユーザー設定
// ---------------------------------------------------------------------
#define USB_PRODUCT_NAME "Sim Pedal + Handbrake"
#define USB_VID          0x2341
#define USB_CUSTOM_PID   0x803A

#define DEBUG_SERIAL     0   // 1: シリアルモニタに生ADC値を出力(キャリブレーション用)

// --- ピン ---
const int PIN_PEDAL_GAS    = 27;  // DE-9 pin2
const int PIN_PEDAL_BRAKE  = 28;  // DE-9 pin3
const int PIN_PEDAL_CLUTCH = 29;  // DE-9 pin4
const int PIN_PEDAL_DETECT = 15;  // DE-9 pin6 (3V3接続, 内蔵プルダウン使用)
const int PIN_HANDBRAKE    = 26;

const int ADC_MAX = 4095;         // 12bit

// --- ペダルキャリブレーション (12bit生値) {離した時, 踏み込んだ時} ---
// 値が逆に動く場合は min と max を入れ替えると反転します。
struct Cal { int minV; int maxV; };
const Cal CAL_GAS    = { 0, ADC_MAX };
const Cal CAL_BRAKE  = { 0, ADC_MAX };
const Cal CAL_CLUTCH = { 0, ADC_MAX };

const unsigned long DETECT_STABLE_MS = 250;  // 接続検出の安定時間

// --- サイドブレーキ (12bit生値の範囲 → 30段階) ---
const float HB_RAW_MIN = 30.0;
const float HB_RAW_MAX = 38.0;
const int   HB_STEPS   = 30;                 // 0〜29
const int   STABLE_THRESHOLD = 5;
const float FAST_MOVE_THRESHOLD = 1.0;

// ---------------------------------------------------------------------
// HIDレポート (16bit x 4軸, 論理範囲 0〜4095)
// ---------------------------------------------------------------------
const uint8_t desc_hid_report[] = {
  HID_USAGE_PAGE ( HID_USAGE_PAGE_DESKTOP     ) ,
  HID_USAGE      ( HID_USAGE_DESKTOP_JOYSTICK ) ,
  HID_COLLECTION ( HID_COLLECTION_APPLICATION ) ,
    HID_USAGE         ( HID_USAGE_DESKTOP_X  ) ,
    HID_USAGE         ( HID_USAGE_DESKTOP_Y  ) ,
    HID_USAGE         ( HID_USAGE_DESKTOP_Z  ) ,
    HID_USAGE         ( HID_USAGE_DESKTOP_RX ) ,
    HID_LOGICAL_MIN   ( 0                    ) ,
    HID_LOGICAL_MAX_N ( 4095, 2              ) ,
    HID_REPORT_COUNT  ( 4                    ) ,
    HID_REPORT_SIZE   ( 16                   ) ,
    HID_INPUT         ( HID_DATA | HID_VARIABLE | HID_ABSOLUTE ) ,
  HID_COLLECTION_END
};

Adafruit_USBD_HID usb_hid;

struct __attribute__((packed)) {
  uint16_t x;   // Gas
  uint16_t y;   // Brake
  uint16_t z;   // Clutch
  uint16_t rx;  // Handbrake
} gp_report;

// ---------------------------------------------------------------------
// ペダル処理
// ---------------------------------------------------------------------
int rawGas = 0, rawBrake = 0, rawClutch = 0;

// ペダル接続検出: HIGHが一定時間続いたら接続とみなす。LOWで即切断。
bool pedalsConnected() {
  static bool connected = false;
  static unsigned long highSince = 0;
  if (digitalRead(PIN_PEDAL_DETECT)) {
    if (highSince == 0) highSince = millis();
    if (!connected && millis() - highSince >= DETECT_STABLE_MS) connected = true;
  } else {
    highSince = 0;
    connected = false;
  }
  return connected;
}

// 生値 → 0〜ADC_MAX (min>max なら反転)
int scaleAxis(int raw, Cal c) {
  long lo = c.minV, hi = c.maxV;
  if (lo == hi) return 0;
  long v = (long)(raw - lo) * ADC_MAX / (hi - lo);
  if (v < 0) v = 0;
  if (v > ADC_MAX) v = ADC_MAX;
  return (int)v;
}

// ---------------------------------------------------------------------
// サイドブレーキ処理
// ---------------------------------------------------------------------
float smoothPot = HB_RAW_MIN;
int   currentStep = 0;
int   pendingStep = 0;
int   stableCount = 0;
float lastRawStep = 0.0;

int updateHandbrakeStep() {
  long sum = 0;
  for (int i = 0; i < 64; i++) sum += analogRead(PIN_HANDBRAKE);
  float currentPot = (float)sum / 64.0;

  smoothPot = (currentPot * 0.15) + (smoothPot * 0.85);

  float rawStep = map(smoothPot * 10.0, HB_RAW_MIN * 10.0, HB_RAW_MAX * 10.0, 0.0, (float)(HB_STEPS - 1));
  int candidate = round(rawStep);

  if (abs(rawStep - lastRawStep) > FAST_MOVE_THRESHOLD) {
    currentStep = candidate;
    pendingStep = candidate;
    stableCount = 0;
  } else {
    if (candidate == pendingStep) {
      stableCount++;
    } else {
      pendingStep = candidate;
      stableCount = 0;
    }
    if (stableCount >= STABLE_THRESHOLD) currentStep = pendingStep;
  }
  lastRawStep = rawStep;

  if (currentStep < 0)            currentStep = 0;
  if (currentStep > HB_STEPS - 1) currentStep = HB_STEPS - 1;
  return currentStep;
}

// ---------------------------------------------------------------------
void setup() {
  analogReadResolution(12);

  TinyUSBDevice.setID(USB_VID, USB_CUSTOM_PID);
  TinyUSBDevice.setProductDescriptor(USB_PRODUCT_NAME);

  usb_hid.setPollInterval(2);
  usb_hid.setReportDescriptor(desc_hid_report, sizeof(desc_hid_report));
  usb_hid.begin();

  pinMode(PIN_PEDAL_DETECT, INPUT_PULLDOWN);
  pinMode(PIN_PEDAL_GAS, INPUT);
  pinMode(PIN_PEDAL_BRAKE, INPUT);
  pinMode(PIN_PEDAL_CLUTCH, INPUT);

  smoothPot = analogRead(PIN_HANDBRAKE);

#if DEBUG_SERIAL
  Serial.begin(115200);
#endif
}

void loop() {
  // --- ペダル ---
  bool connected = pedalsConnected();
  if (connected) {
    rawGas    = analogRead(PIN_PEDAL_GAS);
    rawBrake  = analogRead(PIN_PEDAL_BRAKE);
    rawClutch = analogRead(PIN_PEDAL_CLUTCH);
    gp_report.x = scaleAxis(rawGas,    CAL_GAS);
    gp_report.y = scaleAxis(rawBrake,  CAL_BRAKE);
    gp_report.z = scaleAxis(rawClutch, CAL_CLUTCH);
  } else {
    gp_report.x = gp_report.y = gp_report.z = 0;  // 未接続時は0
  }

  // --- サイドブレーキ ---
  int step = updateHandbrakeStep();
  gp_report.rx = (uint16_t)map(step, 0, HB_STEPS - 1, 0, ADC_MAX);

  // --- USB送信 ---
  if (usb_hid.ready()) {
    usb_hid.sendReport(0, &gp_report, sizeof(gp_report));
  }

#if DEBUG_SERIAL
  static uint32_t lastPrint = 0;
  if (millis() - lastPrint > 100) {
    lastPrint = millis();
    Serial.print("connected="); Serial.print(connected);
    Serial.print(" gas=");      Serial.print(rawGas);
    Serial.print(" brake=");    Serial.print(rawBrake);
    Serial.print(" clutch=");   Serial.print(rawClutch);
    Serial.print(" hb_pot=");   Serial.print(smoothPot);
    Serial.print(" hb_step=");  Serial.println(step);
  }
#endif

  delay(10);
}
