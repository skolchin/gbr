#-------------------------------------------------------------------------------
# Name:        Go board recognition project
# Purpose:     Module definition file
#
# Author:      skolchin
#
# Created:     04.07.2019
# Copyright:   (c) skolchin 2019
#-------------------------------------------------------------------------------

# Register dataset classes
import sys
if sys.version_info[0] < 3:
    from dataset import GrDataset
    from ps_dataset import GrPascalDataset
    from grlog import GrLog
else:
    from gr.dataset import __reg_dataset_fmt
    from gr.ps_dataset import GrPascalDataset
    from gr.grlog import GrLog

#GrLog.init()
GrDataset.addFormat("pascal", GrPascalDataset)



