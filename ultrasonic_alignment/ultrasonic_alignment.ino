#include <Servo.h>
Servo sonic;
 
void setup(){
  sonic.attach(3);
  sonic.write(90); // move servo (sonic) to center position at 90°
} 
void loop(){
 
}
