#!/bin/bash
# 
# This is a script to collect DNS-over-HTTPS traffic
# 
# Configuration
# - Web browser : Firefox with DoH enabled
# - DoH client  : N/A
# - DoH server  : mozilla.cloudflare-dns.com
# 

# How many samples to be collected for each website
HOWMANY_SAMPLES=10

# Firefox and tcpdump time in seconds => about 1 minute for one website
FIREFOX_STANDBY=10
FIREFOX_SURFING=30
FIREFOX_CLOSING=10
TCPDUMP_STANDBY=5
TCPDUMP_CLOSING=5

# Tcpdump packet filter
TCPDUMP_FILTERS='(host 104.16.248.249 or host 104.16.249.249) and (port 443)'

# Enter sudo password
read -sp 'Enter sudo password: ' PASSWORD

# If a pipe exists on stdout, it is for real-time prediction
if [ -p /dev/stdout ]; then
  echo $PASSWORD | sudo -S tcpdump -l -i eth0 $TCPDUMP_FILTERS --immediate-mode
  exit 1
fi

# Prevent firefox showing the safe mode dialog after crash
export MOZ_DISABLE_AUTO_SAFE_MODE=1

mkdir -p ../collection

for ((i=1; i<=HOWMANY_SAMPLES; i++))
do
  # Create a folder for each cycle
  current_date_time="`date +%Y%m%d%H%M%S`"
  mkdir -p ../collection/$current_date_time

  # One cycle of data collection
  while read line
  do
    echo $i $line

    # Start firefox
    firefox &
    sleep $FIREFOX_STANDBY

    # Start traffic collection
    echo $PASSWORD | sudo -S tcpdump -U -i eth0 $TCPDUMP_FILTERS -w ../collection/$current_date_time/$line.pcap &
    sleep $TCPDUMP_STANDBY
    
    # Open a tab in firefox
    firefox --new-tab $line & 
    sleep $FIREFOX_SURFING

    # Stop traffic collection
    echo $PASSWORD | sudo -S pkill -f tcpdump
    sleep $TCPDUMP_CLOSING

    # Close firefox (clean DNS cache)
    pkill -f firefox
    sleep $FIREFOX_CLOSING

    # Clean cache file, browsing histroy, session store ...
    rm -rf ~/.cache/mozilla/firefox/*
    rm -rf ~/.mozilla/firefox/*.default*/*.sqlite
    rm -rf ~/.mozilla/firefox/*.default*/sessionstore*
  done < ../collection/websites.txt

done
    

