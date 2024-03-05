import numpy as np
from matplotlib import pyplot as plt
import librosa
from scipy.signal import windows, convolve


def beatTracker(input_file):
    """
    Wrapper function to call beat tracker
    
    """
    bt = ellisBeatTracker()
    beats, downbeats = bt(input_file)
    return beats, downbeats


class ellisBeatTracker():
    """
    Beat tracker implementation as defined in Ellis, 2007.
    """
    
    def __init__(self, metric=4, plot=False, print_logs=False):
        self.plot = plot
        self.print_logs = print_logs

        # Define stft params
        self.window_time = 0.032
        self.hop_time = 0.004
        self.hop_length=round(8000*self.hop_time)
        self.win_length=round(8000*self.window_time)
        self.frame_rate = 8000/self.hop_length

        # Define meter (defaults to 4/4)
        self.metric = metric

    def __call__(self, input_file):
        # Run beat tracking steps
        self.read_file(input_file)
        self.get_onset_strength()
        self.estimate_tempo()
        self.beats, self.beat_idxs = self.track_beats()
        self.downbeats = self.get_downbeats()

        return self.beats, self.downbeats

    
    def read_file(self, input_file):
        """
        Reads input .wav file 
        """
        snd, rate = librosa.load(input_file, sr=None)
        if self.print_logs:
            print(f"audio size: {snd.shape[0]}, sampling rate: {rate}")
        self.snd = snd
        self.orig_rate = rate

    def get_onset_strength(self):
        """
        Calculate the onset strength at each frame.
        
        """
        # # Resample to 8kHz
        snd_8k = librosa.resample(self.snd, orig_sr=self.orig_rate, target_sr=8000)
        if self.print_logs:
            print(f"audio size after resampling: {snd_8k.shape[0]}")

        # # STFT (32ms window, 4ms hop size)
        stft_mag = np.abs(librosa.stft(snd_8k, 
                                hop_length=self.hop_length, 
                                win_length=self.win_length
                                )
                    )
        if self.plot:
            print(f"Plotting stft spectrogram:")
            librosa.display.specshow(stft_mag, x_axis='time', y_axis='linear')

        # # Map to 40 Melbanks
        melfb = librosa.filters.mel(sr=8000, n_fft=2048, n_mels=40)
        mel = np.dot(melfb, stft_mag)

        # # Mel spectrogram to dB
        mel_db = librosa.amplitude_to_db(mel)

        if self.plot:
            print(f"Plotting mel spectrogram (40 melbanks):")
            librosa.display.specshow(mel_db, y_axis='mel', x_axis='time')


        # apply first order diff
        mel_db_delta = librosa.feature.delta(mel_db)
        # half wave rect
        mel_db_delta[mel_db_delta<0] = 0

        if self.plot:
            print(f"Plotting mel spectrogram (delta):")
            librosa.display.specshow(mel_db_delta, x_axis='time', y_axis='mel')

        # Sum across all freqency bands
        freq_sum = mel_db_delta.sum(axis=0)

        # Define gaussian window
        gauss_size_t = 0.02
        gauss_length = round(mel_db_delta.shape[1]*gauss_size_t)
        gauss_std = 10  # not defined in Ellis, 2007 - determined by comparing against librosa's Onset detection
        gauss_w = windows.gaussian(M=gauss_length, std=gauss_std)
        
        # Apply gaussian filter convolution
        filtered = convolve(freq_sum, gauss_w, mode='same') / sum(gauss_w)
        
        # Normalize by dividing over standard dev
        filtered_norm = filtered/filtered.std()
        if self.plot:
            print(f"Plotting normalized gaussian convoluted signal:")
            plt.plot(filtered_norm[:2800])

        self.onset_env = filtered_norm

    def _tps(self, i):
        """
        TPS function. Takes frame index (i) as param.
        Outputs TPS score at frame i.
        """
        i_time = i/self.frame_rate
        val = self.filtered_norm_corr[i]
        w = np.exp(-0.5*((np.log2(i_time/0.5)/0.9)**2))
        return w*val

    def estimate_tempo(self):
        """
        Estimate the tempo based on onset strength
        """
        # Autocorrelate gaussian filtered signal
        self.filtered_norm_corr = librosa.autocorrelate(self.onset_env)
        if self.plot:
            plt.plot(self.filtered_norm_corr[:round(self.frame_rate*4)])

        # Apply weighted gaussian onto autocorrelation signal
        tps1 = self.filtered_norm_corr.copy()
        tps2 = self.filtered_norm_corr.copy()
        tps3 = self.filtered_norm_corr.copy()
        for i, val in enumerate(self.filtered_norm_corr):
            if i ==0:
                tps1[i] = 0
                tps2[i] = 0
                tps3[i] = 0
            elif i >= 4*self.frame_rate:
                break
            else:
                tps1[i] = self._tps(i)
                tps2[i] = self._tps(i) + 0.5*self._tps(2*i) + 0.25*self._tps(2*i -1) + 0.25*self._tps(2*i + 1)
                tps3[i] = self._tps(i) + 0.33*self._tps(3*i) + 0.33*self._tps(3*i -1) + 0.33*self._tps(3*i + 1)

        combined = tps2+tps3

        if self.plot:
            plt.plot(tps1[:1000])
            plt.plot(tps2[:1000])
            plt.plot(tps3[:1000])
            plt.plot(combined[:1000])

        # Select frame with maximum value as estimated tempo
        tempo_idx = tps1.argmax()
        tempo = 60/(tempo_idx/self.frame_rate)
        if self.print_logs:
            print(f"The estimated tempo is: {tempo} BPM")

        tempo_idx = combined.argmax()
        tempo = 60/(tempo_idx/self.frame_rate)
        if self.print_logs:
            print(f"The estimated tempo (tps2 + tps3) is: {tempo} BPM")
        
        self.tempo = tempo

    # Based on matlab code from paper

    def _beatsimple(self, period, alpha):
        """
        Dynamic programming algorithm for beat tracking.
        Adapted from matlab code defined in the Ellis 2007 paper.
        
        """
        # backlink(time) is best predecessor for this point
        # cumscore(time) is total cumulated score to this point
        backlink = np.zeros(len(self.onset_env))
        cumscore = self.onset_env.copy()

        # Search range for previous beat
        prev_range = range(int(np.round(-2*period)), int(-np.round(period/2)))

        # Log-gaussian window over that range
        txcost = (-alpha*abs((np.log(prev_range/-period))**2))

        if self.print_logs:
            print(f"Starting DP loop...")
        for i in range(-prev_range[-1], len(self.onset_env)):
            timerange = range(i + prev_range.start, i+prev_range.stop)

            # Search over all possible predecessors and apply transition weighting
            scorecands = txcost + cumscore[timerange]
            # Find best predecessor beat
            max_val = scorecands.max()
            max_idx = scorecands.argmax()
            
            # Add on local score
            cumscore[i] = max_val + self.onset_env[i]
            # Store backtrace
            backlink[i] = timerange[max_idx]
            
        if self.print_logs:
            print(f"Starting backtrace...")
        # Start backtrace from best cumulated score
        beat_idx = cumscore.argmax()
        beats = []
        beat_idxs = []

        # then find all its predecessors
        while backlink[beat_idx] > 0:
            beats.append(beat_idx/self.frame_rate)
            beat_idxs.append(beat_idx)
            beat_idx = int(backlink[beat_idx])
            
        return np.flip(beats), np.flip(beat_idxs)


    def track_beats(self):
        period = (60/self.tempo) * self.frame_rate # samples per beat = num seconds between beats * frame_rate
        if self.print_logs:
            print(f"The period is: {period} samples per beat")
        beats, beat_idxs = self._beatsimple(period=period, alpha=680)

        return beats, beat_idxs
    
    def get_downbeats(self):
        """
        Get downbeats based on a known meter and onset strength.
        
        E.g. If we know that the meter is 4/4, 
        then we find the set of beats that are 4-beats apart with the largest combined onset strength.
        """
        # Trim beat indices to exclude first 5 seconds
        beat_idxs_trimmed = np.array(self.beat_idxs)[np.array(self.beat_idxs) >= 5*self.frame_rate]
        beat_onset_strength = self.onset_env[beat_idxs_trimmed]

        # Sum onset strength for each set of beats according to the metric
        # i.e. If meter is 4/4, sum onset strength every 4 beats, for 4 possible down beat positions.
        self.metric_strength = []
        for i in range(self.metric):
            self.metric_strength.append(beat_onset_strength[i::self.metric].sum())


        # Take the position with maximum metric strength as downbeat
        metric_pos = np.array(self.metric_strength).argmax()
        downbeat_idxs = beat_idxs_trimmed[metric_pos::self.metric]

        downbeats = downbeat_idxs/self.frame_rate

        return downbeats
    
