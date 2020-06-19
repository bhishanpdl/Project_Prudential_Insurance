#!/usr/bin/env sh

######################################################################
# @author      : Bhishan (Bhishan@BpMacpro.local)
# @file        : make_html
# @created     : Thursday Jun 18, 2020 10:45:53 EDT
#
# @description : Create html files from notebooks 
######################################################################
rm ../html/*.html
jupyter-nbconvert *.ipynb
mv *.html ../html/
ls ../html/

