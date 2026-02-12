import type {ReactElement} from 'react';
import {Redirect} from '@docusaurus/router';

export default function DownloadPage(): ReactElement {
  return <Redirect to="/downloads" />;
}
