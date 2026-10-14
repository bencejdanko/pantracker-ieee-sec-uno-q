// Pan motor (TB6612FNG Motor B) for the tracker. The Linux side streams set_motor() at >=10 Hz;
// if nothing arrives for WATCHDOG_MS the driver is put in standby, so a crashed/killed Python
// process (or a dropped web jog) can never leave the motor running.
#include <Arduino.h>
#include <Arduino_RouterBridge.h>

#define PIN_STBY 3
#define PIN_PWMB 6
#define PIN_BIN1 9
#define PIN_BIN2 10
// Diagnostic: also drive Motor A (M1) with the same command, to find which terminal the turntable is on.
#define DRIVE_BOTH 1
#define PIN_PWMA 5
#define PIN_AIN1 7
#define PIN_AIN2 8

static const uint32_t WATCHDOG_MS = 200;
static volatile uint32_t last_cmd_ms = 0;
static volatile bool running = false;
static volatile uint32_t watchdog_trips = 0;

static void motor_off() {
    analogWrite(PIN_PWMB, 0);
    digitalWrite(PIN_BIN1, LOW);
    digitalWrite(PIN_BIN2, LOW);
#if DRIVE_BOTH
    analogWrite(PIN_PWMA, 0);
    digitalWrite(PIN_AIN1, LOW);
    digitalWrite(PIN_AIN2, LOW);
#endif
    digitalWrite(PIN_STBY, LOW);
    running = false;
}

static void motor_set(int speed) {
    speed = constrain(speed, -255, 255);
    if (speed == 0) {
        motor_off();
        return;
    }
    digitalWrite(PIN_STBY, HIGH);
    digitalWrite(PIN_BIN1, speed > 0 ? HIGH : LOW);
    digitalWrite(PIN_BIN2, speed > 0 ? LOW : HIGH);
    analogWrite(PIN_PWMB, abs(speed));
#if DRIVE_BOTH
    digitalWrite(PIN_AIN1, speed > 0 ? HIGH : LOW);
    digitalWrite(PIN_AIN2, speed > 0 ? LOW : HIGH);
    analogWrite(PIN_PWMA, abs(speed));
#endif
    running = true;
}

// Every command (including 0) feeds the watchdog.
static bool rpc_set_motor(int speed) {
    last_cmd_ms = millis();
    motor_set(speed);
    return true;
}

static bool rpc_stop() {
    last_cmd_ms = millis();
    motor_off();
    return true;
}

static int rpc_watchdog_trips() {
    return (int)watchdog_trips;
}

void setup() {
    pinMode(PIN_STBY, OUTPUT);
    pinMode(PIN_PWMB, OUTPUT);
    pinMode(PIN_BIN1, OUTPUT);
    pinMode(PIN_BIN2, OUTPUT);
#if DRIVE_BOTH
    pinMode(PIN_PWMA, OUTPUT);
    pinMode(PIN_AIN1, OUTPUT);
    pinMode(PIN_AIN2, OUTPUT);
#endif
    motor_off();

    Bridge.begin();
    Bridge.provide_safe("set_motor", rpc_set_motor);
    Bridge.provide_safe("stop", rpc_stop);
    Bridge.provide_safe("watchdog_trips", rpc_watchdog_trips);
}

void loop() {
    if (running && (millis() - last_cmd_ms) > WATCHDOG_MS) {
        motor_off();
        watchdog_trips++;
    }
    delay(5);
}
