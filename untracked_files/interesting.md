● Now I can see it clearly:                                                                                                                                                                                
                                    
  - USB speaker (UACDemoV1.0) hardware device [13]: stereo, 48000Hz, I16 only                                                                                                                              
  - USB mic (USB PnP Sound Device) hardware device [3]: mono, 44100Hz or 48000Hz, I16 only                                                                                                                 
   
  But cpal's "Default Audio Device" reports ch=2 rate=44100 fmt=F32 — which is ALSA's default wrapper lying about what it supports. The fix is to skip the default device and pick the USB devices directly
   by name.
