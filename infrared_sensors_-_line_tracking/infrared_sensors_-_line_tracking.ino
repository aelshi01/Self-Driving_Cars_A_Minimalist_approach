//Infrared Sensor, defining the left, right and middle sensors
#define IR_R !digitalRead(10)
#define IR_M !digitalRead(4)
#define IR_L !digitalRead(2)
// Defining L298N motor driver 
#define EA 5
#define EB 6
#define I1 7
#define I2 8
#define I3 9
#define I4 11

// Car speed variable set 
#define Speed 150
 
void forward(){
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, LOW);
  digitalWrite(I2, HIGH);
  digitalWrite(I3, HIGH);
  digitalWrite(I4, LOW);
  Serial.println("go forward!");
}
 
void back(){
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, HIGH);
  digitalWrite(I2, LOW);
  digitalWrite(I3, LOW);
  digitalWrite(I4, HIGH);
  Serial.println("go back!");
}
 
void left(){
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, LOW);
  digitalWrite(I2, HIGH);
  digitalWrite(I3, LOW);
  digitalWrite(I4, HIGH);
  Serial.println("go left!");
}
 
void right(){
  analogWrite(EA, Speed);
  analogWrite(EB, Speed);
  digitalWrite(I1, HIGH);
  digitalWrite(I2, LOW);
  digitalWrite(I3, HIGH);
  digitalWrite(I4, LOW); 
  Serial.println("go right!");
} 
 
void stop(){
   digitalWrite(EA, LOW);
   digitalWrite(EB, LOW);
   Serial.println("Stop!");
} 
 
void setup(){
  Serial.begin(9600);
  pinMode(IR_R,INPUT);
  pinMode(IR_M,INPUT);
  pinMode(IR_L,INPUT);
}
 
void loop() {
  if(IR_M){
    forward();
  }
  else if(IR_R) { 
    right();
    while(IR_R);                             
  }   
  else if(IR_L) {
    left();
    while(IR_L);  
  }
}
