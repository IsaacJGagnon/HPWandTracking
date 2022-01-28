#include <SoftwareSerial.h>

SoftwareSerial BT(4,5);
int IRpin = 3;

int ForwardCode[9] = {6100, 580, 1600, 580, 1600, 580, 1600, 580, 710};
int ReverseCode[9] = {6100, 600, 1550, 650, 1550, 650, 650, 1500, 1600};
int StopCode[9] = {6000, 650, 1500, 700, 600, 1550, 650, 1600, 1450};
int WhistleCode[9] = {6000, 700, 1500, 700, 1450, 750, 550, 1600, 600};
int MuteCode[9] = {6100, 600, 1550, 650, 650, 1550, 1550, 650, 650};
int SteamCode[9] = {6050, 650, 1550, 600, 650, 1550, 1550, 650, 1500};
int LightCode[9] = {6050, 650, 1550, 650, 650, 1500, 700, 1500, 650};

char command;

void setup() {
  BT.begin(9600);
  Serial.begin(9600);
}

void IRsetup(void) {
  pinMode(IRpin, OUTPUT);
  digitalWrite(IRpin, LOW);
}

void IRcarrier(unsigned int IRtimemicroseconds)
{
  int i;
  for(i=0; i<(IRtimemicroseconds / 26); i++)
  {
    digitalWrite(IRpin, HIGH);
    delayMicroseconds(9);
    digitalWrite(IRpin, LOW);
    delayMicroseconds(9);
  }
}

void IRsendCode(size_t sz, int *code)
{
  for(int i = 0; i < sz; i++)
  {
    if ( (i%2) == 0)
    {
      IRcarrier(code[i]);
    } else
    {
      delayMicroseconds(code[i]);
    }
  }
}

void loop() {
  if (BT.available())
  {
    command = (BT.read());
    if (command == '1')
    {
      Serial.println("LOCOMOTOR");
      IRsendCode((sizeof(ForwardCode)/sizeof(ForwardCode[0])),ForwardCode);
    }if (command == '2'){
      Serial.println("ARRESTO MOMENTUM");
      IRsendCode((sizeof(StopCode)/sizeof(StopCode[0])),StopCode);
    }else if (command == '3'){
      Serial.println("SILENCIO");
      IRsendCode((sizeof(MuteCode)/sizeof(MuteCode[0])),MuteCode);
    } else if (command == '4'){
      Serial.println("FLIPENDO");
      //IRsendCode((sizeof(ForwardCode)/sizeof(ForwardCode[0])),ForwardCode);
    } else {
      Serial.println("Unkown");
    }
  }
}
