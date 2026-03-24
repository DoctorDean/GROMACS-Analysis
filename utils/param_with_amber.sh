#!/bin/bash 

#Usage: param_with_amber.sh <3-letter ligand name>

# Activate the pyenv environment for AmberTools
# Assuming 'ambertools' is the name of your environment
eval "$(pyenv init -)"
pyenv activate ambertools

NEW_LIG_NAME=$1

# Protonate the ligand with Open Babel 
obabel "${NEW_LIG_NAME}.pdb" -O "${NEW_LIG_NAME}_h.pdb" -h

# Ensure CONECT records are removed from PDB file 
sed -i '/CONECT/d' "${NEW_LIG_NAME}_h.pdb"

# Run antechamber to generate a mol2 file 
antechamber -i "${NEW_LIG_NAME}_h.pdb" -fi pdb -o "${NEW_LIG_NAME}.mol2" -fo mol2 -c bcc -s 2

# Check that the QM/MM calculation was successful
if tail -n 4 sqm.out | grep -q "Calculation Completed"; then
	echo "Calculation Completed Successfully. MOL2 Genenerated"
	# Successful calculation means we can test if all the parameters we require are available.
	parmchk2 -i "${NEW_LIG_NAME}.mol2" -f mol2 -o "${NEW_LIG_NAME}.frcmod"

	# Create file for tleap to parameterise the ligand
    cat > tleap.in <<EOF
source oldff/leaprc.ff99SB
source leaprc.gaff
${NEW_LIG_NAME} = loadmol2 ${NEW_LIG_NAME}.mol2
check ${NEW_LIG_NAME}
loadamberparams ${NEW_LIG_NAME}.frcmod
saveoff ${NEW_LIG_NAME} ${NEW_LIG_NAME}.lib
saveamberparm ${NEW_LIG_NAME} ${NEW_LIG_NAME}.prmtop ${NEW_LIG_NAME}.rst7
quit
EOF
	# Run tleap with the updated input file
    tleap -f tleap.in

    # Convert AMB2GMX
    acpype -p "${NEW_LIG_NAME}.prmtop" -x "${NEW_LIG_NAME}.rst7"

else
	echo "Calculation may not have been completed. Check sqm.out"
fi

# Deactivate the pyenv environment
pyenv deactivate