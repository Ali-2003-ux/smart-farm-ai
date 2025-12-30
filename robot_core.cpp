#include "thingProperties.h"
#include <Servo.h>

void setup() {
  Serial.begin(9600);
  delay(1500);
  initProperties();
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);
  setDebugMessageLevel(2);
  ArduinoCloud.printDebugInfo();
}

void loop() {
  ArduinoCloud.update();
  // Fire fighting logic v2.1
  // Enterprise Logic: Check Watchdog Timer
}
