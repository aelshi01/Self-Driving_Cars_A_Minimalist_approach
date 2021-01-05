 
#include <Servo.h>  //servo library
Servo sonic;      // create servo object (sonic) to control servo motor
 
int Echo = A4;  
int Trig = A5; 
 
#define EA 5
#define EB 6
#define I1 7
#define I2 8
#define I3 9
#define I4 11
#define Speed 220
int rightDistance = 0, leftDistance = 0, middleDistance = 0;
 
void forward(){
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, LOW);
  digitalWrite(I2, HIGH);
  digitalWrite(I3, HIGH);
  digitalWrite(I4, LOW); 
  Serial.println("Forward");
}
 
void back() {
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, HIGH);
  digitalWrite(I2, LOW);
  digitalWrite(I3, LOW);
  digitalWrite(I4, HIGH);
  Serial.println("Back");
}
 
void left() {
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, LOW);
  digitalWrite(I2, HIGH);
  digitalWrite(I3, LOW);
  digitalWrite(I4, HIGH); 
  Serial.println("Left");
}
 
void right() {
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, HIGH);
  digitalWrite(I2, LOW);
  digitalWrite(I3, HIGH);
  digitalWrite(I4, LOW);
  Serial.println("Right");
}
 
void stop() {
  digitalWrite(EA, LOW);
  digitalWrite(EB, LOW);
  Serial.println("Stop!");
} 
 
//Measuring Ultrasonic distance 
int Distance_test() {
  digitalWrite(Trig, LOW);   
  delayMicroseconds(2);
  digitalWrite(Trig, HIGH);  
  delayMicroseconds(20);
  digitalWrite(Trig, LOW);   
  float Fdistance = pulseIn(Echo, HIGH);  
  Fdistance= Fdistance / 58;       
  return (int)Fdistance;
}  
 
void setup() { 
  sonic.attach(3);  // attach servo motor (sonic) on pin 3 to servo object
  Serial.begin(9600);     
  pinMode(Echo, INPUT);    
  pinMode(Trig, OUTPUT);  
  pinMode(I1, OUTPUT);
  pinMode(I2, OUTPUT);
  pinMode(I3, OUTPUT);
  pinMode(I4, OUTPUT);
  pinMode(EA, OUTPUT);
  pinMode(EB, OUTPUT);
  stop();
} 
 
void loop() { 
    sonic.write(90);  //set sonic (servo motor) position according to scaled value
    delay(500); 
    middleDistance = Distance_test();
 
    if(middleDistance <= 20) {     
      stop();
      delay(500);                         
      sonic.write(10);          
      delay(1000);      
      rightDistance = Distance_test();
      
      delay(500);
      sonic.write(90);              
      delay(1000);                                                  
      sonic.write(180);              
      delay(1000); 
      leftDistance = Distance_test();
      
      delay(500);
      sonic.write(90);              
      delay(1000);
      if(rightDistance > leftDistance) {
        right();
        delay(360);
      }
      else if(rightDistance < leftDistance) {
        left();
        delay(360);
      }
      else if((rightDistance <= 20) || (leftDistance <= 20)) {
        back();
        delay(180);
      }
      else {
        forward();
      }
    }  
    else {
        forward();
    }                     
}
