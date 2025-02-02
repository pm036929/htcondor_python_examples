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

FROM jupyter/minimal-notebook:latest
ARG HTCONDOR_VERSION=8.9

USER root

RUN apt-get update \
 && apt-get -y install --no-install-recommends gnupg git vim less \
 && wget -qO - https://research.cs.wisc.edu/htcondor/ubuntu/HTCondor-Release.gpg.key | apt-key add - \
 && echo "deb  http://research.cs.wisc.edu/htcondor/ubuntu/${HTCONDOR_VERSION}/bionic bionic contrib" >> /etc/apt/sources.list \
 && apt-get -y update \
 && apt-get -y install htcondor \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

COPY binder/condor_config.local /etc/condor/condor_config.local
COPY binder/entrypoint.sh ${HOME}/.entrypoint.sh
COPY /Examples ${HOME}/Examples
RUN pip install --no-cache-dir htcondor==${HTCONDOR_VERSION}.* nbstripout \
 && pip install plotly \
 && pip install numpy \
 && nbstripout `find ${HOME}/Examples -name '*.ipynb'` \
 && chmod +x ${HOME}/.entrypoint.sh \
 && chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}/Examples
ENTRYPOINT ["/home/jovyan/.entrypoint.sh"]
CMD ["jupyter", "lab"]
