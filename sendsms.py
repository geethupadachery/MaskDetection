#import requests
#import json
#
#def sendSMS(msg,numbers):
#    headers = {
#    "authkey": "9005A9HoIdi4rny5fc9e1a4P123",
#    "Content-Type": "application/json"
#    }
#
#    data = "{ \"sender\": \"GTURES\", \"route\": \"4\", \"country\": \"1\", \"sms\": [ { \"message\": \""+msg+"\", \"to\": "+json.dumps(numbers)+" } ] }"
#
#    requests.post("https://api.msg91.com/api/v2/sendsms", headers=headers, data=data)
##sendSMS("demo-pg",[9899710306])

from twilio.rest import Client

def sendSMS(msg):

    account_sid = 'ACb92f7c92704b8663bdf1df90998714b4' 
    auth_token = '7175a8a508e7361685c835f5fafeceff' 
    client = Client(account_sid, auth_token) 
    
    message = client.messages.create( 
                                from_='+12142257395',  
                                body=msg,      
                                to='+19899710306' 
                            )
    print(message.sid)

#    print(message.sid)
