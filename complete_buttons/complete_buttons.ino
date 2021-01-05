//defining connections to the L298N H-Bridge controller.
// Numbers correspond to the pin on the Arduino.
#define EA 5 // motor 1
#define EB 6 // motor 2
#define I1 7
#define I2 8
#define I3 9
#define I4 11
#define light 13


unsigned char Speed = 220; //the speed when we are moving forward or backwards
bool state = LOW; //initialising starting state as off
char read_string;
 
void forward(){ 
  digitalWrite(EA,HIGH);
  digitalWrite(EB,HIGH);
  digitalWrite(I1,HIGH);
  digitalWrite(I2,LOW);
  digitalWrite(I3,LOW);
  digitalWrite(I4,HIGH);
  Serial.println("Forward");
}
 
void back(){
  digitalWrite(EA,HIGH);
  digitalWrite(EB,HIGH);
  digitalWrite(I1,LOW);
  digitalWrite(I2,HIGH);
  digitalWrite(I3,HIGH);
  digitalWrite(I4,LOW);
  Serial.println("Back");
}
 
void left(){
  analogWrite(EA,Speed);
  analogWrite(EB,Speed);
  digitalWrite(I1,LOW);
  digitalWrite(I2,HIGH);
  digitalWrite(I3,LOW);
  digitalWrite(I4,HIGH); 
  Serial.println("Left");
}
 
void right(){
  analogWrite(EA,Speed);
  analogWrite(EB,Speed);
  digitalWrite(I1,HIGH);
  digitalWrite(I2,LOW);
  digitalWrite(I3,HIGH);
  digitalWrite(I4,LOW);
  Serial.println("Right");
}
 
void stop(){
  digitalWrite(EA,LOW);
  digitalWrite(EB,LOW);
  Serial.println("Stop!");
}
 
void stateChange(){
  state = !state;
  digitalWrite(light, state);
  Serial.println("Light");  
}
 
void setup() { 
  Serial.begin(9600);
  pinMode(light, OUTPUT); 
  pinMode(I1,OUTPUT);
  pinMode(I2,OUTPUT);
  pinMode(I3,OUTPUT);
  pinMode(I4,OUTPUT);
  pinMode(EA,OUTPUT);
  pinMode(EB,OUTPUT);
  stop();
}
 
void loop() { 
  read_string = Serial.read();
  switch(read_string){
    case 'b': forward(); break;
    case 'f': back();   break;
    case 'l': left();   break;
    case 'r': right();  break;
    case 's': stop();   break;
    case 'a': stateChange(); break;
    default:  break;
  }
}
