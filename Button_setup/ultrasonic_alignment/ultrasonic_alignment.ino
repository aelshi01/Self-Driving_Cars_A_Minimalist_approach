#include <Servo.h>
Servo servo;
 
void setup(){
  servo.attach(3);
  servo.write(90); // move servo to center position at 90°
} 
void loop(){
 
}
