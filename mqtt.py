import sys
import paho.mqtt.client as mqtt 

server     = "maqiatto.com"
port       = 1883

username   = "jaybhavsar98@gmail.com"
password   = "jaybhavsar98"
topic      = "jaybhavsar98@gmail.com/autobot"

message = "1"

if len(sys.argv) > 1:
    message = sys.argv[1]

# The callback for when the client is connected to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result: " + mqtt.connack_string(rc))

    # publish message to the topic
    client.publish(topic, message)

# The callback for when the client is disconnected from the broker
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection")

# The callback for when the client has published a message
def on_publish(client, userdata, mid):
    print("Message published: " + message)

    # message published, disconnect
    client.disconnect()

# create a new MQTT client
client = mqtt.Client()

# configure CA cert for SSL connection

# set username and password
client.username_pw_set(username, password)

# assign callbacks
client.on_connect    = on_connect
client.on_disconnect = on_disconnect
client.on_publish    = on_publish

# start connection
client.connect(server, port)

# let the client loop (this is a blocking call), will unblock when the client disconnects
client.loop_forever()
