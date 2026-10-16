// Over-the-air update: `POST /ota` on port 80 with the app binary as the body (make fw-ota). The
// image goes to the inactive slot, which boots next; the device reboots once the reply is out.
#pragma once

void ota_http_start(void);
