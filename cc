1A client server based prg  TCP to find if the number entered is prime

tcpClientPrime.java
import java.net.*;import java.io.*;
class tcpClientPrime{
public static void main(String args[]){try{Socket cs = new Socket("LocalHost",8001);
BufferedReader infu = new BufferedReader(new InputStreamReader(System.in));
System.out.println("Enter a number : ");int a = Integer.parseInt(infu.readLine());
DataOutputStream out = newDataOutputStream(cs.getOutputStream());out.writeInt(a);
DataInputStream in = new DataInputStream(cs.getInputStream());System.out.println(in.readUTF()); cs.close();	}
catch(Exception e){System.out.println(e.toString());}}}
-------------
import java.net.*;import java.io.*;
class tcpServerPrime{public static void main(String args[]){
try{ServerSocket ss = new ServerSocket(8001);System.out.println("Server Started...............");
Socket s = ss.accept();DataInputStream in = new
DataInputStream(s.getInputStream()); int x= in.readInt();DataOutputStream otc = new
DataOutputStream(s.getOutputStream()); int y = x/2;if(x ==1 || x ==2 || x ==3)
{otc.writeUTF(x + "is Prime");System.exit(0);}for(int i=2; i<=y; i++){if(x%i != 0){
otc.writeUTF(x + " is Prime");}else{otc.writeUTF(x + " is not Prime");}}}
catch(Exception e){System.out.println(e.toString());}}}

=======================================================================================================================================

1B A client server TCP based chatting application

ChatClient.java
import java.net.*;import java.io.*;
class ChatClient{public static void main(String args[]){
try {Socket s = new Socket("Localhost", 8000);BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
DataOutputStream out = new DataOutputStream(s.getOutputStream());DataInputStream in = new DataInputStream(s.getInputStream());
String msg;System.out.println("To stop chatting with server type STOP");System.out.print("Client Says: ");
while ((msg = br.readLine()) != null) {out.writeBytes(msg + "\n");if (msg.equals("STOP")) {break;}
System.out.println("Server Says : " + in.readLine());System.out.print("Client Says : ");}
br.close();in.close();out.close();s.close();} catch (Exception e) {e.printStackTrace();}}}
-------------------------
ChatServer.java
import java.net.*;import java.io.*;
class ChatServer {public static void main(String args[]) {try 
{ServerSocket ss = new ServerSocket(8000);System.out.println("Waiting for client to connect..");Socket s = ss.accept();
BufferedReader br = new BufferedReader(new InputStreamReader(System.in));DataOutputStream out = new DataOutputStream(s.getOutputStream());
DataInputStream in = new DataInputStream(s.getInputStream());String receive, send;while ((receive = in.readLine()) != null) 
{if (receive.equals("STOP")) {break;}System.out.println("Client Says : " + receive);System.out.print("Server Says : ");
send = br.readLine();out.writeBytes(send + "\n");}
br.close();in.close();out.close();s.close();} catch (Exception e) {e.printStackTrace();}}}
=======================================================================================================================================
2A Client Server communication model using UDP(even or odd)

udpClientEO.java
import java.io.*;import java.net.*;
public class udpClientEO{public static void main(String args[]){try
{DatagramSocket ds = new DatagramSocket(1000);BufferedReader br = new BufferedReader(newInputStreamReader(System.in));
System.out.println("Enter a number : ");String num = br.readLine();byte b[] = new byte[1024];b=num.getBytes();
DatagramPacket dp = newDatagramPacket(b,b.length,InetAddress.getLocalHost(),2000);ds.send(dp);
byte b1[] = new byte[1024];DatagramPacket dp1 = newDatagramPacket(b1,b1.length); ds.receive(dp1);String str = new String(dp1.getData(),0,dp1.getLength());
System.out.println(str);}catch(Exception e){e.printStackTrace();}}}

udpServerEO.java
import java.io.*;import java.net.*;public class udpServerEO
{public static void main(String args[]){try{DatagramSocket ds = new DatagramSocket(2000);byte b[] = new byte[1024];
DatagramPacket dp = new DatagramPacket(b,b.length);ds.receive(dp);String str = new String(dp.getData(),0,dp.getLength());
System.out.println(str);int a= Integer.parseInt(str);String s= new String();if (a%2 == 0)s = "Number is even"; else
s = "Number is odd"; byte b1[] = new byte[1024];b1 = s.getBytes();DatagramPacket dp1 = new
DatagramPacket(b1,b1.length,InetAddress.getLocalHost(),1000);ds.send(dp1);}catch(Exception e){e.printStackTrace();}}}

==============================================================================================================================================
2B Client Server communication model using UDP(Factorial)

import java.io.*;import java.net.*;public class 

ClientFact
{public static void main(String args[]){try{DatagramSocket ds = new DatagramSocket(1000);
BufferedReader br = new BufferedReader(newInputStreamReader(System.in));System.out.println("Enter a number : ");String num = br.readLine();
byte b[] = new byte[1024];b=num.getBytes();DatagramPacket dp = new DatagramPacket(b,b.length,InetAddress.getLocalHost(),2000);ds.send(dp);
byte b1[] = new byte[1024];DatagramPacket dp1 = new DatagramPacket(b1,b1.length);ds.receive(dp1);String str = newString(dp1.getData(),0,dp1.getLength());System.out.println(str);
}catch(Exception e){e.printStackTrace();}}}













































