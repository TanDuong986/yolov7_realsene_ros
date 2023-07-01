#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def pub_txt():
    rospy.init_node('talker',anonymous=False)
    pub = rospy.Publisher('chatbot',String,queue_size=10)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ =="__main__":
    try:
        pub_txt()
    except rospy.ROSInterruptException:
        pass