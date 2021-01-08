'''
Remove unnecessary files after a Ray Tune run to save disk space.
'''

import argparse
import os
import glob

def cleanup(strTuneDir):
    lsFoldDirs = glob.glob(os.path.join(strTuneDir, 'fold*'))
    lsFoldDirs.sort()
    for strFoldDir in lsFoldDirs:
        lsModelDirs = glob.glob(os.path.join(strFoldDir, 'train*'))
        lsModelDirs.sort()
        for strModelDir in lsModelDirs:
            lsCheckpointDirs = glob.glob(os.path.join(strModelDir, 'checkpoint*'))
            lsCheckpointDirs.sort()
            # Remove all checkpoint weights except the last
            if len(lsCheckpointDirs) > 1:
                for strCheckpointDir in lsCheckpointDirs[:-1]:
                    lsWeights = glob.glob(os.path.join(strCheckpointDir, '*.h5'))
                    for strWeight in lsWeights:
                        os.remove(strWeight)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tunedir', type=str, help='tune directory to clean up')                       
    args = parser.parse_args()
    cleanup(args.tunedir)
