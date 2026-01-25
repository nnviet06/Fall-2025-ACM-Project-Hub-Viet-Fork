#include <Servo.h>

Servo myServo;
Servo myServo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;

void setup() 
{
  Serial.begin(9600);
  myServo.attach(9);       // keep attached so it can hold when powered
  myServo2.attach(10);
  servo3.attach(11);
  servo4.attach(12);
  servo5.attach(13);
  servo6.attach(14);
}

void loop() 
{
  if (Serial.available()) 
  {
    // Read format: "90,45,30,60,80,90\n"
    int angle1 = Serial.parseInt();
    int angle2 = Serial.parseInt();
    int angle3 = Serial.parseInt();
    int angle4 = Serial.parseInt();
    int angle5 = Serial.parseInt();
    int angle6 = Serial.parseInt();
    
    // Validate all angles
    if (angle1 >= 0 && angle1 <= 180) myServo.write(angle1);
    if (angle2 >= 0 && angle2 <= 180) myServo2.write(angle2);
    if (angle3 >= 0 && angle3 <= 180) servo3.write(angle3);
    if (angle4 >= 0 && angle4 <= 180) servo4.write(angle4);
    if (angle5 >= 0 && angle5 <= 180) servo5.write(angle5);
    if (angle6 >= 0 && angle6 <= 180) servo6.write(angle6);
    
    // Clear buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}