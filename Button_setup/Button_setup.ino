#define light 13    //light is assigned a number 13
bool state = LOW; //The initial state is off
char read_string;      //Bluetooth character defined in a function
 
void setup() {
  pinMode(light, OUTPUT);
  Serial.begin(9600);
}
 
//Control LED sub function
void stateChange() {
  state = !state; 
  digitalWrite(light, state);  
}
 
void loop() {
  //Function defined to recieve the data from Bluetooth serial port
  read_string = Serial.read();
  if(read_string == 'a'){
    stateChange();
  }
}
