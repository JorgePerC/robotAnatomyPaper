#!/usr/bin/awk -f

BEGIN {
    print "timestamp,theta,dist_mm,quality"
	
}

/^Scan received at time:/ {
    timestamp = $5 +0
    next
}

/theta:/ {
    sub(/^S/, "", $0)
    print timestamp"," $2"," $4"," $6 
    next
}

