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
--------------------
udpServerEO.java
import java.io.*;import java.net.*;public class udpServerEO
{public static void main(String args[]){try{DatagramSocket ds = new DatagramSocket(2000);byte b[] = new byte[1024];
DatagramPacket dp = new DatagramPacket(b,b.length);ds.receive(dp);String str = new String(dp.getData(),0,dp.getLength());
System.out.println(str);int a= Integer.parseInt(str);String s= new String();if (a%2 == 0)s = "Number is even"; else
s = "Number is odd"; byte b1[] = new byte[1024];b1 = s.getBytes();DatagramPacket dp1 = new
DatagramPacket(b1,b1.length,InetAddress.getLocalHost(),1000);ds.send(dp1);}catch(Exception e){e.printStackTrace();}}}

==============================================================================================================================================
2B Client Server communication model using UDP(Factorial)

udpClientFact.java
import java.io.*;import java.net.*;public class udpClientFact
{public static void main(String args[]){try{DatagramSocket ds = new DatagramSocket(1000);
BufferedReader br = new BufferedReader(newInputStreamReader(System.in));System.out.println("Enter a number : ");String num = br.readLine();
byte b[] = new byte[1024];b=num.getBytes();DatagramPacket dp = new DatagramPacket(b,b.length,InetAddress.getLocalHost(),2000);ds.send(dp);
byte b1[] = new byte[1024];DatagramPacket dp1 = new DatagramPacket(b1,b1.length);ds.receive(dp1);String str = newString(dp1.getData(),0,dp1.getLength());System.out.println(str);
}catch(Exception e){e.printStackTrace();}}}
----------------------------------
udpServerFact.java
import java.io.*;import java.net.*;public class udpServerFact
{public static void main(String args[]){try{DatagramSocket ds = new DatagramSocket(2000);byte b[] = new byte[1024];
DatagramPacket dp = new DatagramPacket(b,b.length);ds.receive(dp);String str = newString(dp.getData(),0,dp.getLength());
System.out.println(str);int a= Integer.parseInt(str);int f = 1, i;String s= new String();for(i=1;i<=a;i++){f=f*i;}
s=Integer.toString(f);String str1 = "The Factorial of " + str + " is : " +f; byte b1[] = new byte[1024]; b1 =str1.getBytes();
DatagramPacket dp1 = new DatagramPacket(b1,b1.length,InetAddress.getLocalHost(),1000);
ds.send(dp1);}catch(Exception e){e.printStackTrace();}}}
===============================================================================================================================
2C calculator operations like addition,subtraction, multiplication and division
RPCClient.java
import java.io.*;import java.net.*;class RPCClient {RPCClient() {try {InetAddress ia = InetAddress.getLocalHost();
DatagramSocket ds = new DatagramSocket();DatagramSocket ds1 = new DatagramSocket(1300);System.out.println("\nRPC Client\n");
System.out.println("Enter method name and parameter like add 3 4\n");while (true) {BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
String str = br.readLine();byte b[] = str.getBytes();DatagramPacket dp = new DatagramPacket(b, b.length, ia, 1200);ds.send(dp);
dp = new DatagramPacket(b, b.length);ds1.receive(dp);String s = new String(dp.getData(), 0, dp.getLength());if (s.contains("e")) {
System.out.println("Program exited");System.exit(0);} else {System.out.println("\nResult = " + s + "\n");continue;}}} catch (Exception e) {
e.printStackTrace();}}public static void main(String[] args) {new RPCClient();}}
---------------------------------
RPCServer.java
import java.util.*;import java.net.*;class RPCServer {DatagramSocket ds;DatagramPacket dp;
String str, methodName, result;int val1, val2;RPCServer() {try {ds = new DatagramSocket(1200);byte b[] = new byte[4096];
System.out.println("waiting for client instruction");while (true) {dp = new DatagramPacket(b, b.length);ds.receive(dp);
str = new String(dp.getData(), 0, dp.getLength());if (str.equalsIgnoreCase("q")) {System.out.println("Program stop request received....");
String flag = "exit";byte b1[] = flag.getBytes();DatagramSocket ds1 = new DatagramSocket();DatagramPacket dp1 = new DatagramPacket(b1, b1.length, InetAddress.getLocalHost(), 1300);
ds1.send(dp1);System.out.println("Program stop request processed");System.exit(0);} else {StringTokenizer st = new StringTokenizer(str, " ");
int i = 0;while (st.hasMoreTokens()) {String token = st.nextToken();methodName = token;val1 = Integer.parseInt(st.nextToken());
val2 = Integer.parseInt(st.nextToken());}}System.out.println(str);InetAddress ia = InetAddress.getLocalHost();if (methodName.equalsIgnoreCase("add")) {result = "" + add(val1, val2);
} else if (methodName.equalsIgnoreCase("sub")) {result = "" + sub(val1, val2);} else if (methodName.equalsIgnoreCase("mul")) {
result = "" + mul(val1, val2);} else if (methodName.equalsIgnoreCase("div")) {result = "" + div(val1, val2);}
byte b1[] = result.getBytes();DatagramSocket ds1 = new DatagramSocket();DatagramPacket dp1 = new DatagramPacket(b1, b1.length, InetAddress.getLocalHost(), 1300);
System.out.println("result : " + result + "\n");ds1.send(dp1);}} catch (Exception e) {e.printStackTrace();}}
public int add(int val1, int val2) {return val1 + val2;}public int sub(int val3, int val4) {return val3 - val4;}
public int mul(int val3, int val4) {return val3 * val4;}public int div(int val3, int val4) {return val3 / val4;}
public static void main(String[] args) {new RPCServer();}}
=====================================================================================================================================
2D square, square root, cube and cube root
RPCNumClient.java
import java.io.*;import java.net.*;class RPCNumClient{RPCNumClient(){try{InetAddress ia = InetAddress.getLocalHost();DatagramSocket ds = new DatagramSocket();
DatagramSocket ds1 = new DatagramSocket(1300);System.out.println("\nRPC Client\n");System.out.println("1. Square of the number - square\n2. Square root of the number - squareroot\n3. Cube of the number - cube\n4. Cube root of the number -cuberoot");
System.out.println("Enter method name and the number\n");while (true){BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
String str = br.readLine();byte b[] = str.getBytes();DatagramPacket dp = new DatagramPacket(b,b.length,ia,1200);ds.send(dp);
dp = new DatagramPacket(b,b.length);ds1.receive(dp);String s = new String(dp.getData(),0,dp.getLength());System.out.println("\nResult = " + s + "\n");
}}catch (Exception e){e.printStackTrace();}}public static void main(String[] args){new RPCNumClient();}}
-------------------------------
RPCNumServer.java
import java.util.*;import java.net.*;import java.io.*;class RPCNumServer{DatagramSocket ds;DatagramPacket dp;String str,methodName,result;
int val;RPCNumServer(){try{ds=new DatagramSocket(1200);byte b[]=new byte[4096];while(true){dp=new DatagramPacket(b,b.length);ds.receive(dp);
str=new String(dp.getData(),0,dp.getLength());if(str.equalsIgnoreCase("q")) {System.exit(1);}else{StringTokenizer st = new StringTokenizer(str," ");
int i=0;while(st.hasMoreTokens()){String token=st.nextToken();methodName=token;val = Integer.parseInt(st.nextToken());}}System.out.println(str);
InetAddress ia = InetAddress.getLocalHost();if(methodName.equalsIgnoreCase("square")){result= "" + square(val);}else if(methodName.equalsIgnoreCase("squareroot"))
{result= "" + squareroot(val);}else if(methodName.equalsIgnoreCase("cube")){result= "" + cube(val);}else if(methodName.equalsIgnoreCase("cuberoot")){
result= "" + cuberoot(val);}byte b1[]=result.getBytes();DatagramSocket ds1 = new DatagramSocket();DatagramPacket dp1 = newDatagramPacket(b1,b1.length,InetAddress.getLocalHost(), 1300);
System.out.println("result :"+result+"\n"); ds1.send(dp1);}}catch (Exception e){e.printStackTrace();}}public double square(int a) throws Exception
{double ans;ans = a*a;return ans;}public double squareroot(int a) throws Exception{double ans;ans = Math.sqrt(a);return ans;}public double cube(int a) throws Exception
{double ans;ans = a*a*a;return ans;}public double cuberoot(int a) throws Exception{double ans;ans = Math.cbrt(a);return ans;}public static void main(String[] args){new RPCNumServer();}}
==========================================================================================================================================
3 multicast Socket example
BroadcastClient.java
import java.net.*;import java.io.*;public class BroadcastClient {
public static final int PORT = 1234;public static void main(String args[]) throws Exception {MulticastSocket socket;DatagramPacket packet;
InetAddress address;//set the mulitcast address to your local subnetaddress = InetAddress.getByName("239.1.2.3");socket = new MulticastSocket(PORT);//join a Multicast group and wait for a message
socket.joinGroup(address);byte[] data = new byte[100];packet = new DatagramPacket(data, data.length);for (;;) {// receive the packets
socket.receive(packet);String str = new String(packet.getData());System.out.println("Message received from " + packet.getAddress() + " Message is : " + str);} // for} // main} // end BroadcastClient
----------------------------------------------------
BroadcastServer.java
import java.net.*;import java.io.*;import java.util.*;public class BroadcastServer {public static final int PORT = 1234;
public static void main(String args[]) throws Exception {MulticastSocket socket;DatagramPacket packet;InetAddress address;// set the multicast address to your local subnet
address = InetAddress.getByName("239.1.2.3");socket = new MulticastSocket();// join a Multicast group and send the group messagessocket.joinGroup(address);byte[] data = null;
for (;;) {Thread.sleep(10000);System.out.println("Sending ");String str = ("This is Pravin Calling ... ");data= str.getBytes();
packet = new DatagramPacket(data, str.length(), address, PORT);
// Sends the packetsocket.send(packet);} // end for} // end main} // end class BroadcastServer
====================================================================================================================================
4A  show the object communication using RMI
ClientDate.java
import java.rmi.*;import java.io.*;public class ClientDate{public static void main(String args[]) throwsException {String s1;InterDate h1 = (InterDate)Naming.lookup("DS");
s1 = h1.display();System.out.println(s1);}}//javac ServerDate.java//javac ClientDate.java//rmic ServerDate//rmiregistry//java ServerDate(another cmd)//java ClientDate(another cmd)
---------------------------
InterDate.java
InterDate.java
import java.rmi.*;public interface InterDate extends Remote{public String display() throws Exception;}
---------------------------
ServerDate.java
import java.rmi.*;import java.rmi.server.*;import java.util.*;public class ServerDate extends UnicastRemoteObject implements
InterDate {public ServerDate() throws Exception{}public String display() throws Exception{String str = "";Date d = new Date();str = d.toString();return str;
}public static void main(String args[]) throwsException {ServerDate s1 = new ServerDate();Naming.bind("DS",s1);System.out.println("Object registered.....");}}
==========================================================================================================================================
4B RMI based application program that converts digits to words
ClientConvert.java
import java.rmi.*;import java.io.*;public class ClientConvert{public static void main(String args[]) throwsException {InterConvert h1 =
(InterConvert)Naming.lookup("Wrd"); BufferedReaderbr = new BufferedReader(newInputStreamReader(System.in));System.out.println("Enter a number :\t"); String no = br.readLine();
String ans = h1.convertDigit(no);System.out.println("The word representation of the entered digit is : " +ans);}}
//javac ServerConvert.java//javac ClientConvert.java//rmiregistry//java ServerConvert(another cmd)//java ClientConvert(another cmd)
--------------------------------------------
InterConvert.java
import java.rmi.*;public interface InterConvert extends Remote{public String convertDigit(String no) throws Exception;}
---------------------------------------------
ServerConvert.java
import java.rmi.*;import java.rmi.server.*;public class ServerConvert extends UnicastRemoteObject implementsInterConvert {
public ServerConvert() throws Exception{}public String convertDigit(String no) throws Exception{String str = "";for(int i = 0; i < no.length(); i++)
{int p = no.charAt(i);if( p == 48){str += "zero ";}if( p == 49){str += "one ";}if( p == 50){str += "two ";}if( p == 51){str += "three ";}
if( p == 52){str += "four ";}if( p == 53){str += "five ";}if( p == 54){str += "six ";}if( p == 55){str += "seven ";}if( p == 56){str += "eight ";}
if( p == 57){str += "nine ";}}return str;}public static void main(String args[]) throwsException {ServerConvert s1 = new ServerConvert();
Naming.bind("Wrd",s1);System.out.println("Object registered....");}}
===============================================================================================================================

Practical 5A: Implementing “Big” Web Service.
1) Creating a Web Service
A. Choosing a Container:1. choose File > New Project. Select Web Application from the Java Web.2. Name the project CalculatorWSApplication. Select a location for the project. Click 3. Select your server and Java EE version and click Finish. Next.
B. Creating a Web Service from a Java Class 1. Right-click the CalculatorWSApplication node and choose New > Web Service.2. Name the web service CalculatorWS and type org.me.calculator in Package. Leave
Create Web Service from Scratch selected. If you are creating a Java EE 6 project on GlassFish or WebLogic, select Implement Web Service as a Stateless Session Bean
3. Click Finish. The Projects window displays the structure of the new web service and the source code is shown in the editor area.
2) Adding an Operation to the Web Service 
A. To add an operation to the web service:1. Change to the Design view in the editor 2. Click Add Operation in either the visual designer or the context menu. The Add Operation dialog opens.
3. In the upper part of the Add Operation dialog box, type add in Name and type int in the Return Type drop-down list 4. In the lower part of the Add Operation dialog box, click Add and create aparameter of type int named i.
5. Click Add again and create a parameter of type int called j. You now seethe following:6. Click OK at the bottom of the AddOperation dialog box. Youreturn to the
7. The visual designer now displays the following:8. Click Source. And code the following. 
@WebMethod(operationName = "add")
publicint add(@WebParam(name = "i") inti, @WebParam(name = "j") int j){int k = i + j;return k;}
3) Deploying and Testing the Web Service
A. To test successful deployment to a GlassFish or WebLogic server:
1. Right-click the project and choose Deploy. The IDE starts the application server, builds the application, and deploys the application to the server
2. In the IDE's Projects tab, expand the Web Services node of theCalculatorWSApplication project. Right-click the CalculatorWS node, and chooseTest Web Service.
3. The IDE opens the tester page in your browser, if you deployed a web applicationto the GlassFish server.
4. If you deployed to the GlassFish server, type two numbers in the tester page,as shown below:5. The sum of the two numbers is displayed:
4) Consuming the Web Service
Now that you have deployed the web service, you need to create a client to make use ofthe web service's add method.
1. Client: Java Class in Java SE Application 1. Choose File > New Project. Select Java Application from the Java category. Name theproject CalculatorWS_Client_Application. Leave Create Main Class selected andaccept all other default settings. Click Finish.
2. Right-click the CalculatorWS_Client_Application node and choose New > WebService Client. The New Web Service Client wizard opens.
3. Select Project as the WSDL source. Click Browse. Browse to the CalculatorWS webservice in the CalculatorWSApplication project. When you have selected the webservice, click OK.
4. Do not select a package name. Leave this field empty5. Leave the other settings at default and click Finish. The Projects window displays thenew web service client, with a node for the add method that you created:
6. Double-click yourthe Source Editor.main() method.main class so that it opens in Drag the add node below the
You now see the following:
public static void main(String[] args)
{
// TODO code application logic here
}
private static int add(inti, int j)
{
org.me.calculator.CalculatorWS_Service service = new
org.me.calculator.CalculatorWS_Service();
org.me.calculator.CalculatorWS port = service.getCalculatorWSPort();
return port.add(i, j);
}
7. In the main() method body, replace the TODO comment with code that initializes
values for i and j, calls add(), and prints the result.
public static void main(String[] args)
{
inti = 3;
int j = 4;
int result = add(i, j);
System.out.println("Result = " + result);
}
8. Surround the main() method code with a try/catch block that prints an exception.
public static void main(String[] args)
{
try
{
inti = 3;
int j = 4;
int result = add(i, j);
System.out.println("Result = " + result);
} catch (Exception ex) {
System.out.println("Exception: " + ex);
}
}
9. Right-click the project node and choose Run.
The Output window now shows the sum:
compile:
run:
Result = 7
BUILD SUCCESSFUL (total time: 1 second)
============================================================================================================================
Practical 5B: Implementing Web Service that connects to MySQL database.
Building Web Service:-
JAX-WS is an important part of the Java EE platform.
JAX-WS simplifies the task of developing Web services using Java technology.
It provides support for multiple protocols such as SOAP, XML and by providing a facility for
supporting additional protocols along with HTTP.
With its support for annotations, JAX-WS simplifies Web service development and reduces
the size of runtime files.
Here basic demonstration of using IDE to develop a JAX-WS Web Service is given.
After creating the web service, create web service clients that use the Web service over a
network which is called consuming a web service.
The client is a servlet in a web application.
Let’s build a Web Service that returns the book name along with its cost for a particular ISBN.
To begin building this service, create the data store. The server will access the data stored in
a MySQL table to serve the client.
2) Creating MySQL DB Table
create database bookshop;
use bookshop;
Create a table named Books that will store valid books information
create table books(isbn varchar(20) primary key, bookname varchar(100), bookprice
varchar(10));
Insert valid records in the Books table
insert into books values("111-222-333","Learn My SQL","250");
insert into books values("111-222-444","Java EE 6 for Beginners","850");
insert into books values("111-222-555","Programming with Android","500");
insert into books values("111-222-666","Oracle Database for you","400");
insert into books values("111-222-777","Asp.Net for advanced programmers","1250");
2) Creating a web service
i. Choosing a container
Web service can be either deployed in a Web container or in an EJB container.
If a Java EE 6 application is created, use a Web container because EJBs can be placed
directly in a Web application.
ii. Creating a web application
To create a Web application, select File - New Project.
New Project dialog box appears. Select Java Web available under the Categories
section and Web Application available under the Projects section. Click Next.
New Web Application dialog box appears. Enter BookWS as the project name in the
Project Name textbox and select the option Use Dedicated Folder for Storing Libraries.
Click Next. Server and Settings section of the New Web Application dialog box
appears. Choose the default i.e. GlassFish v3 Domain as the Web server, the Java EE 6
Web as the Java EE version and the Context Path.
Click –Finish
The Web application named BookWS is created.
iii. Creating a web service
Right-click the BookWS project and select New -> Web Service as shown in diagram
New Web Service dialog box appears. Enter the name BookWS in the Web Service
Name textbox, webservice in the Package textbox, select the option Create Web
Service from scratch and also select the option implement web service as a stateless
session bean as shown in the diagram.
Click Finish.
The web service in the form of java class is ready.
3) Designing the web service
Now add an operation which will accept the ISBN number from the client to the
web service.
i. Adding an operation to the web service
Change the source view of the BookWS.java to design view by clicking
Design available just below the name of the BookWS.java tab.
The window changes as shown in the diagram.
Click Add Operation available in the design view of the web service.
Add Operation dialog appears. Enter the name getBookDetails in the Name textbox and
java.lang.String in the Return Type textbox as shown in the diagram.
In Add Operation dialog box, click Add and create a parameter of the type
String named isbn as shown in the diagram.
Click Ok. The design view displays the operation added as shown in the diagram.
Click Source. The code spec expands due to the operation added to the web service
as shown in the diagram.
Modify the code spec of the web service BookWS.java.

Code Spec
packagewebservice;
importjava.sql.*;
importjavax.jws.WebMethod;
importjavax.jws.WebParam;
importjavax.jws.WebService;
importjavax.ejb.Stateless;
@WebService()
@Stateless()
public class BookWS {
/**
* Web service operation
*/
@WebMethod(operationName = "getBookDetails") public
String getBookDetails(@WebParam(name = "isbn")
String isbn) {
//TODO write your implementation code here
Connection dbcon = null;
Statement stmt = null;
ResultSetrs = null;
String query = null;
try
{
Class.forName("com.mysql.jdbc.Driver").newInstance();
dbcon =
DriverManager.getConnection("jdbc:mysql://localhost/bookshop","root","123");
stmt = dbcon.createStatement();
query = "select * from books where isbn = '" +isbn+
"'"; rs = stmt.executeQuery(query); rs.next();
String bookDetails = "<h1>The name of the book is <b>"
+rs.getString("bookname") + "</b> and its cost is <b>" +rs.getString("bookprice") +
"</b></h1>.";
returnbookDetails;
}
catch(Exception e)
{
System.out.println("Sorry failed to connect to the database.." + e.getMessage());
}
return null;
}
4) Adding the MySQL connector
We need to add a reference of MySQL connector to our web service. It is via
this connector that our web service will be able to communicate with the
database.
Right click on the libraries and select Add JAR/Folder as shown in the diagram.
Choose the location where mysql-coonector-java-5.1.10-bin is located, select it
and click on open as shown.
5) Deploying and testing the web service
When a web service is deployed to a web container, the IDE allows testing the
web service to see if it functions as expected.
The tester application provided by GlassFish, is integrated into the IDE for this
purpose as it allows the developer to enter values and test them.
No facility for testing whether an EJB module is deployed successfully is
currently available.
To test the BookWS application, right click the BookWS project and select Deploy
as shown in the diagram.
The IDE starts the server, builds the application and deploys the application to
the server.
Follow the progress of these operations in the BookWS (run-deploy) and GlassFish
v3 Domain tabs in the Output view.
Now expand the web services directory of the BookWS project, right-click the
BookWS Web service and select Test web service as shown in the diagram.
The IDE opens the tester page in the web browser, if the web application is
deployed using GlassFish server as shown in the figure.
Enter the ISBN number as shown in the diagram.
Click getBookDetails. The book name and its cost are displayed as shown in
the diagram.
6) Consuming the web service
Once the web service is deployed, the next most logical step is to create a client
to make use of the web service’s getBookDetails() method.
i. Creating a web application
To create a web application, select File -> New Project.
New project dialog box appears, select java web available under the categories section
and web application available under the projects section. Click Finish.
New web application dialog box appears. Enter BookWSServletClient as the project
name in the Project Name textbox and select the option Use Dedicated Folder for
Storing Libraries.
Click Next. Server and settings section of the new web application, dialog box appears.
Choose the default i.e. GlassFish v3 Domain as the web serevr, the Java EE 6 web as
the Java EE version and the context path.
Click Finish.
The web application named BookWSServletClient is created.
ii. Adding the web service to the client application
Right-click the BookWSServletClient project and select New -> Web Service Client
as shown in the diagram.
New Web Service Client dialog box appears. In the Project section, click Browse and
browse through the web service which needs to be consumed. Click ok. The name of
the web service appears in the New Web Service Client as shown in the diagram.
Leave the other settings as it is. Click Finish
The Web Service Reference directory is added to the BookWSServletClient application
as shown in the diagram. It displays the structure of the newly created client including
the getBookDetails() method created earlier.

iii. Creating a servlet
Create retreiveBookDetails.java using NetBeans IDE.
Right click source package directory, select New -> Servlet.
New Servlet dialog box appears. Enter retreiveBookDetails in the Class Name
textbox and enter servlet in the package textbox.
Click Next. Configure Servlet Deployment section of the New Servlet dialog
box appears. Keep the defaults.
Click Finish.
This creates the servlet named retreiveBookDetails.java in the servlet package.
retreiveBookDetails.java is available with the default skeleton created by the
NetBeans IDE which needs to be modified to hold the application logic.
In the retreieveBookDetails.java source code, remove the following
comments available in the body of the processRequest() method.
/*TODO output your page here*/
Replace the following code spec:
out.println("<h1>Servlet retreiveBookDetails at " + request.getContextPath ()
+ "</h1>");
With the code spec of the getBookDetails() operation of the web service by
dragging and dropping the getBookDetails operation as shown in the diagram.
The Servlet code spec changes as shown in the diagram
The web service is instantiated by the @WebServiceRef annotation.
Now change the following code spec:
java.lang.Stringisbn = “”;
to
java.lang.Stringisbn = request.getParameter(“isbn”);
iv. Creating an HTML form
Once the web service is added and the servlet is created, the form to accept ISBN
from the user needs to be coded.
Since NetBeans IDE by default [as a part of Web Application creation] makes
available index.jsp file. Modify it to hold the following code spec.
<%@page contentType="text/html" pageEncoding="UTF8"%><!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01
Transitional//EN"
"http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF8"><title>SOAP Cleint - Get Book Details</title>
</head>
<body bgcolor="pink">
<form name="frmgetBookDetails" method="post"
action="retreiveBookDetails"><h1>
ISBN : <input type="text"
name="isbn"/><br><br></h1>
<input type="submit"
value="Submit"/></form>
</body>
</html>
v. Building the Web Application
Build the web application.
Right click BookWSServletClient project and select Build.
Once the Build menu item is clicked the details about the compilation and building of
the BookWSServletClient Web application appears in the output –
BookWSServletClient (dist) window.
vi. Running the Application
Once the compilation and building of the web application is done run the application.
Right click the BookWSServerCleint project and select run.
Once the run processing completes in NetBeans IDE a web browser is automatically
launched and the BookWSServletCleint application is executed as shown in the diagram.
Enter the ISBN as shown in the diagram
Click Submit. The book name and its cost are displayed as shown in the diagram.

===================================================================================================================================
#Implement Windows Hyper V virtualization

1.First we have to uninstall vmware software if already installed on computer because the VMware Workstation installer does not support 
running on a Hyper-V virtual machine. after uninstalling vmware we can proceed to next step go to control panel and click on uninstall a 
program.
2.Click on Turn windows features on or off.
Now in windows features check on Hyper-V option.
3.After Restart Search for hyper-v manager in search box and click on that.
4.for creating virtual machine first we have to create virtual switch click on virtual switch manager option.
5.Select External as a connection type and then click on create virtual switch.
Create new Virtual switch and install windows XP .iso and virtual machine will start.
=======================================================================================================================================
#Develop application for Microsoft Azure.
Step 1:
To develop an application for Windows Azure on Visual Studio install the “Microsoft Azure SDK for .NET (VS 2010) – 2.8.2.1”
Step2:
Turn windows Features ON or OFF:
Go to Control panel and click on programs.
Turn Windows features on or off.
Step3:
Now, Start the visual studio 2010 and Go To File->New->Project
Expand Visual C#-> Select Cloud
Give project name as azure
select webrole
click on ok 
In sultion explorer=>Right Click on WebRole1>>ADD>>New Item
Add a New web Form. Give it a name. Click Add
Deploy the project:
Select start debugging 
then click on run google chrome.
==============================================================================================================================================
#Develop application for Google App Engine

Open Eclipse Luna. Go to Help Menu Install New Software…
 In Install window Click on the “Add” button besides the Work with textbox. Add Repository window appears. Enter the Location as 
“https://dl.google.com/eclipse/plugin/4.4” and click on “OK” button.
 From the available softwares select the required softwares and tools as shown in the below image for the GAE. Then click on the “Next” button.
 In the Install Details window click on “Next” button.
 In the Next Window "Review the Items to be Installed" then click on “Next”
 In the next window for Review Licenses select the option “I accept……” and click on “Finish”
 button.
 After Installation you will get option to "Restart Eclipse", click on Yes. So that the software you selected gets updated...
 Now, go to File Menu_New_Other.
-In the New window select Google_Web Application Project and click on “Next” button.
-Enter the details for the new Web application project. Deselect the Use Google Web Toolkit option under the section Google SDKs. 
-Click on the “Finish” button
-From the Package Explorer open the .java file (Here it is “Google_App_EngineServlet.java”).
-Edit the file as required (Unedited file too can be used. Here the editing is done to “what should be displayed” on the browser). 
-Save the file. 
-Click on the Run option available on the Tools bar.
-In the browser (Here, Google Chrome) type the address as “localhost:8888” which is "Default".
-In localhost:8888 the link to the Google_App_EngineServlet.java file as Google_App_Engine is displayed. Click on this link. 
-It will direct you to “localhost:8888/Google_App_Engine”.
-The output text entered in the java program is displayed as the output when clicked the link “Google_App_Engine”.
=============================================================================================================================
