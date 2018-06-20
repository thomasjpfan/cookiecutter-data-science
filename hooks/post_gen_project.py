#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

{% if cookiecutter.include_test_csv == "False" %}

os.remove('data/raw/train.csv')
os.remove('data/raw/test.csv')

{% endif %}
