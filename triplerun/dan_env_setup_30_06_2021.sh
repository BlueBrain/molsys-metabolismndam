
# Autogen file to load NGVBASE environment and setup neurodamus-py in dev mode
NGVBASE=/gpfs/bbp.cscs.ch/project/proj62/software/30-06-2021


NGVDEVHOME=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
NEURODAMUS_PYTHON=$NGVDEVHOME/neurodamus-py

module purge
echo "Sourcing $NGVBASE/env_setup.sh"
source $NGVBASE/env_setup.sh

if [ ! -d "$NGVDEVHOME/venv" ]; then
  echo "No venv found. Creating python virtual environment ..."
  module load python
  python3 -m venv $NGVDEVHOME/venv
fi

. $NGVDEVHOME/venv/bin/activate

if [ ! -d "$NGVDEVHOME/neurodamus-py" ]; then
  echo "Cloning neurodamus-py from \$NGVBASE/.npy -> $NGVBASE/.npy" 
        git clone $NGVBASE/.npy $NGVDEVHOME/neurodamus-py
	( cd $NGVDEVHOME/neurodamus-py; git remote set-url origin ssh://$USER@bbpcode.epfl.ch/sim/neurodamus-py )
  echo "Setting up dev mode: pip install -e $NGVDEVHOME/neurodamus-py"
  pip install -e $NGVDEVHOME/neurodamus-py
fi

PYTHONPATH=$(p=$(echo $PYTHONPATH | tr ":" "\0" | grep -v -z $PY_NEURODAMUS_ROOT | tr "\0" ":"); echo ${p%:})

echo "neurodamus-py is now in $(pip show neurodamus-py | grep Location)"
echo "Base NGV setup: \$NGVBASE=$NGVBASE"
echo "Dev NGV setup: \$NGVDEVHOME=$NGVDEVHOME"
cd $NGVDEVHOME
echo "Done."

get_latest_mod() {
    if [ ! -d "$NGVDEVHOME/_ngv_mods" ]; then
        echo "Copying neocortex mods into \$NGVDEVHOME/_ngv_mods from \$NEURODAMUS_NEOCORTEX_ROOT/lib/mod ..." 
        cp -r $NEURODAMUS_NEOCORTEX_ROOT/lib/mod $NGVDEVHOME/_ngv_mods
    chmod -R ugo+w $NGVDEVHOME/_ngv_mods
    else
        echo "_ngv_mods folder already exists!"
    fi
}

build_custom_mod() {
    (cd $NGVDEVHOME; rm -rf ./x86_64; build_neurodamus.sh _ngv_mods)
}

