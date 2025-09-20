# Making the stochasticChannelLayer

from scipy.stats import truncnorm
from scipy.stats import uniform

r = 4           # For upsampling -> number of complex samples per symbol
roll_off = 0.35 # Roll off factor
L = 31          # Number of taps (odd) for RRC filter
f_s = 2e6
T_bound = 1/f_s # Go through the resharch paper Deep Learning Based Communication Over the Air  (content under table 1) 
time_delay = np.random.uniform(-1,1) # To convert the time delay into discrete domain, time dilay is giving relative to the sampling period 
CFO = 5e3
CFO_std = CFO/f_s
snr = snr


# function to create the complex values
def real_to_complex_tensor(inp_tensor):
  if len(inp_tensor.shape) == 0 or inp_tensor.shape[0] % 2 != 0:
    raise ValueError("Input tensor must have an even number of elements.")
  if len(inp_tensor.shape) == 1:
    inp_tensor = tf.reshape(inp_tensor, [-1, 2])
  real_part = inp_tensor[:, 0]
  imag_part = inp_tensor[:, 1]
  complex_tensor = tf.complex(real_part, imag_part)
  return complex_tensor

def complex_to_real_tensor(inp_tensor):
   real_part , imag_part = tf.math.real(inp_tensor), tf.math.imag(inp_tensor)
   real_part = tf.reshape(real_part,[-1,1])
   imag_part = tf.reshape(imag_part,[-1,1])
   return tf.reshape(tf.concat([real_part,imag_part],1),[-1])

# Upsample
def upsampling(inp,r):
  #complex_tensor = real_to_complex_tensor(inp)
  com_reshape = tf.reshape(inp,[-1,1])
  zeros_vec = np.zeros((com_reshape.shape[0],r-1),np.float32)
  zeros_tensor =  tf.complex(zeros_vec,zeros_vec)
  upsampled = tf.concat([com_reshape,zeros_tensor],1)
  return tf.reshape(upsampled,[-1])

# Normalized RRC with time shift
def NRRC_filter(num_taps, roll_off, time_delay):
  t = np.linspace(-(num_taps-1)/2,(num_taps-1)/2,num_taps) - time_delay
  eps = np.finfo(float).eps # Small epsilon to avoid divisiomn by zero
  pi = np.pi
  def RRC_filter_coff(t):
    if abs(t) < eps:  # For t==0
      return 1.0 - roll_off + (4*roll_off/pi)
    elif roll_off != 0 and (abs(t-1/(4*roll_off))<eps or abs(t+1/(4*roll_off))<eps):
      return (roll_off/np.sqrt(2))*(1 + 2/pi)*np.sin(pi/(4*roll_off)) + (1- 2/pi)*np.cos(pi/(4*roll_off))
    else:
      nu = np.sin(pi*t*(1-roll_off)) + 4*roll_off*t*np.cos(pi*t*(1+roll_off))
      den = pi*t*(1-(4*roll_off*t)**2)
      return nu/(den + eps)
  filter_coff = np.array([RRC_filter_coff(T) for T in t])
  NRRC_filter_coff = filter_coff / np.sum(np.abs(filter_coff))
  print(f"Time_delay = {time_delay}")
  plt.stem(t,NRRC_filter_coff)  # Plot for visualization
  return tf.constant(NRRC_filter_coff,dtype = tf.float32)

# Phase offset
def PhaseOffset_vec(inp_shape,r,CFO_std):
  if inp_shape[0] != None: 
    l = inp_shape[0]
  else:
     l = inp_shape[1]
  CFO_off = truncnorm.rvs(-1.96,1.96)*CFO_std  # boundaries will be selected for 95% confidence
                                               # CFO_min and CFO_max (boundaries) will be selected for 95% confidence
  exp_vec = []
  phase_off = 0
  for i in range(l):
    if i%r ==0:
        phase_off = uniform.rvs(scale = 2*np.pi)
    exp_vec.append(tf.math.exp(tf.constant([0+(2*np.pi*i*CFO_off+phase_off)*1j],dtype=tf.complex64)))
  return tf.reshape(tf.stack(exp_vec),inp_shape)
   

class UpsamplingLayer(keras.layers.Layer):
    def _init_(self, r =r):
        super()._init_()
        self.r = r
    def call(self,inputs):
       return upsampling(inputs,self.r)
    
class PulseShaping(keras.layers.Layer): # The input size will not be changed
    def _init_(self,num_taps,roll_off,time_delay):
      super()._init_()
      self.nrrc_filter = NRRC_filter(num_taps,roll_off,time_delay)
      self.nrrc_filter = tf.reshape(self.nrrc_filter,[num_taps,1,1])
    def call(self, inputs):
      inp_shape = inputs.shape[0]
      real_part , imag_part = tf.math.real(inputs), tf.math.imag(inputs)
      real_part = tf.reshape(real_part,[1,inp_shape,1])
      imag_part = tf.reshape(imag_part,[1,inp_shape,1])
      real_conv = tf.nn.conv1d(real_part,self.nrrc_filter,stride=1,padding="SAME")
      imag_conv = tf.nn.conv1d(imag_part,self.nrrc_filter,stride=1,padding="SAME")
      real_conv = tf.reshape(real_conv,[-1])
      imag_conv = tf.reshape(imag_conv,[-1])
      return tf.complex(real_conv,imag_conv)

class PhaseOffset(keras.layers.Layer):
    def _init_(self,r,CFO_std):
      super()._init_()
      self.r = r
      self.CFO_std = CFO_std
    def call(self,inputs):
       return inputs * PhaseOffset_vec(inputs.shape,self.r,self.CFO_std)

class StochasticChannelLayer(keras.layers.Layer):
    """This channel will output 1D tensor.
        r ----------> upsampling constant (number of complex samples per symbol)
        time_delay -> uniformly distributed time delay between (-1,1), discrete domain, 
                      time dilay is giving relative to the sampling period
        CFO_std ----> CFO_frequency / sampling_frequency is taken as the standared deviation
        snr --------> snr for AWGN channel"""
    def _init_(self, r,time_delay,CFO_std,snr):
        super()._init_()
        self.UpSamplingLayer_inst = UpsamplingLayer(r)
        L = 31          # Number of taps
        roll_off = 0.35 # Roll off factor
        self.PulseShaping_inst = PulseShaping(L,roll_off,time_delay)
        self.PhaseOffset_inst = PhaseOffset(r,CFO_std)
        self.AWGNlayer = keras.layers.GaussianNoise(stddev = np.sqrt(1/10**(snr/10)))
    def call(self, inputs):
      inputs = tf.reshape(inputs,[-1])
      inputs = real_to_complex_tensor(inputs)
      x = self.UpSamplingLayer_inst(inputs)
      x = self.PulseShaping_inst(x)
      x = self.PhaseOffset_inst(x)
      x = self.AWGNlayer(x)
      x = complex_to_real_tensor(x)
      print(x.shape)
      return x