void setup(){
  Serial.begin(9600); // same baud rate (bits per second) for proper serial communication
  pinMode(LED_BUILTIN, OUTPUT); // LED built-in at digital pin 13 and ground pin besides it  
}

// run the below command in pi terminal to list out all the ports with the beginning of "tty"
// ls /dev/tty* 
// this will display all available ports
// now on attaching Arduino via USB to Raspberry Pi, a new port appears i.e., port for Arduino
// note the new port name for Python code in RPi

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    printf(data);
    if (data=="Panel is unclean") {
    digitalWrite(LED_BUILTIN, HIGH); 
    Serial.println("Done");       
    }
    else {
    digitalWrite(LED_BUILTIN, LOW);
    }
  }    
}
