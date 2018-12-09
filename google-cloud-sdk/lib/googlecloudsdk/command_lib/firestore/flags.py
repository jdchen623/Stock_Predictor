# -*- coding: utf-8 -*- #
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flags and helpers for the firestore related commands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.calliope import arg_parsers


def AddCollectionIdsFlag(parser):
  """Adds flag for collection ids to the given parser."""
  parser.add_argument(
      '--collection-ids',
      metavar='COLLECTION_IDS',
      type=arg_parsers.ArgList(),
      help="""
      A list specifying which collections will be included in the operation.
      When omitted, all collections are included.

      For example, to operate on only the 'customers' and 'orders'
      collections:

        $ {command} --collection-ids='customers','orders'
      """)
